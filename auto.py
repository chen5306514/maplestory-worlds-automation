#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MapleStory Worlds 優化自動化系統
使用 YOLO 模型進行智能物件偵測和自動化操作
版本: 2.0
作者: AI Assistant
"""

import cv2
import mss
import numpy as np
import pyautogui
import time
import os
import sys
import ctypes
import logging
import yaml
import threading
import queue
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk, messagebox

try:
    from pynput import keyboard
except Exception:
    keyboard = None

try:
    if sys.platform == "darwin":
        import AppKit
        import Foundation
    else:
        AppKit = None
except Exception:
    AppKit = None

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('auto_system.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

TARGET_CLASSES = {"木面怪人", "石面怪人"}

if sys.platform == "darwin" and AppKit is not None:
    class NativeOverlayView(AppKit.NSView):
        """macOS 原生透明覆盖层视图；背景保持 alpha=0"""
        snapshot = None

        def drawRect_(self, rect):
            try:
                bounds = self.bounds()
                width = bounds.size.width
                height = bounds.size.height
                AppKit.NSColor.clearColor().set()
                AppKit.NSBezierPath.fillRect_(bounds)

                yellow = AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(
                    1.0, 0.83, 0.0, 1.0
                )
                border = AppKit.NSBezierPath.bezierPathWithRect_(
                    Foundation.NSMakeRect(1, 1, width - 2, height - 2)
                )
                border.setLineWidth_(2)
                yellow.set()
                border.stroke()

                snapshot = self.snapshot
                if snapshot is None:
                    self._draw_text(
                        "等待识别...", 12, 16,
                        AppKit.NSFont.boldSystemFontOfSize_(12),
                        AppKit.NSColor.whiteColor()
                    )
                    return

                for detection in snapshot.detections:
                    x1, y1, x2, y2 = detection.bbox
                    x1 = x1 / snapshot.capture_scale_x
                    y1 = y1 / snapshot.capture_scale_y
                    x2 = x2 / snapshot.capture_scale_x
                    y2 = y2 / snapshot.capture_scale_y
                    is_target = detection.class_name in TARGET_CLASSES
                    color = AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(
                        0.0, 1.0, 0.4, 1.0
                    ) if is_target else AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(
                        0.53, 0.60, 0.67, 1.0
                    )
                    path = AppKit.NSBezierPath.bezierPathWithRect_(
                        Foundation.NSMakeRect(x1, height - y2, x2 - x1, y2 - y1)
                    )
                    path.setLineWidth_(2)
                    color.set()
                    path.stroke()
                    self._draw_text(
                        f"{detection.class_name} {detection.confidence:.2f}",
                        x1 + 2, max(8, y1 - 8),
                        AppKit.NSFont.boldSystemFontOfSize_(10),
                        color
                    )

                state = "已暂停" if snapshot.paused else (
                    "攻击" if snapshot.targets else "待机"
                )
                status_text = (
                    f"FPS {self._current_fps} | {state} | "
                    f"截图 {snapshot.capture_ms:.0f}ms | "
                    f"推理 {snapshot.inference_ms:.0f}ms"
                )
                background = AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(
                    0.0, 0.0, 0.0, 0.72
                )
                bg_path = AppKit.NSBezierPath.bezierPathWithRect_(
                    Foundation.NSMakeRect(0, 0, width, 24)
                )
                background.set()
                bg_path.fill()
                self._draw_bottom_bar_text(
                    status_text, 10,
                    AppKit.NSFont.boldSystemFontOfSize_(10),
                    AppKit.NSColor.whiteColor()
                )
            except Exception:
                pass

        def _draw_text(self, text, top_left_x, top_left_y, font, color):
            height = self.bounds().size.height
            attributes = {
                AppKit.NSFontAttributeName: font,
                AppKit.NSForegroundColorAttributeName: color
            }
            value = Foundation.NSString.stringWithString_(text)
            size = value.sizeWithAttributes_(attributes)
            point = Foundation.NSMakePoint(
                top_left_x,
                height - top_left_y - size.height
            )
            value.drawAtPoint_withAttributes_(point, attributes)

        def _draw_bottom_bar_text(self, text, x, font, color):
            attributes = {
                AppKit.NSFontAttributeName: font,
                AppKit.NSForegroundColorAttributeName: color
            }
            value = Foundation.NSString.stringWithString_(text)
            size = value.sizeWithAttributes_(attributes)
            y = max(2.0, (24.0 - size.height) / 2.0)
            value.drawAtPoint_withAttributes_(
                Foundation.NSMakePoint(x, y), attributes
            )
else:
    NativeOverlayView = None

@dataclass
class Detection:
    """偵測結果數據類"""
    bbox: List[int]
    confidence: float
    class_id: int
    class_name: str
    center: Tuple[int, int]
    distance_from_center: float = 0.0

@dataclass
class TargetTrack:
    """跨幀怪物目標；hits 用於確認，misses 用於容忍短暫遮擋"""
    id: int
    detection: Detection
    hits: int = 1
    misses: int = 0
    last_seen: float = 0.0

    @property
    def foot_center(self) -> Tuple[int, int]:
        bbox = self.detection.bbox
        return ((bbox[0] + bbox[2]) // 2, bbox[3])

@dataclass
class VisionSnapshot:
    """视觉线程输出的最新检测结果；控制线程只读取这份快照"""
    created_at: float
    detections: List[Detection]
    targets: List[Detection]
    capture_ms: float = 0.0
    inference_ms: float = 0.0
    capture_scale_x: float = 1.0
    capture_scale_y: float = 1.0
    paused: bool = False

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> Dict:
        """載入配置文件"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            else:
                logger.warning(f"配置文件 {self.config_path} 不存在，使用默认配置")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """獲取默認配置"""
        return {
            'model': {
                'default_path': 'weights/best.pt',
                'confidence_threshold': 0.6,
                'iou_threshold': 0.45,
                'tiled_inference': {
                    'enabled': False,
                    'size': 640,
                    'stride': 256,
                    'exclude_regions': []
                }
            },
            'window': {
                'default': {'left': 100, 'top': 100, 'width': 1200, 'height': 800}
            },
            'controls': {
                'pickup_key': 'z',
                'interact_key': 'space',
                'attack_method': 'click'
            },
            'automation': {
                'action_delay': 0.3,
                'scan_interval': 0.1,
                'max_detection_distance': 200,
                'priority_targets': ['item', 'mob', 'npc']
            },
            'safety': {
                'enable_failsafe': True,
                'max_runtime_hours': 2
            }
        }
    
    def get(self, key_path: str, default=None):
        """獲取配置值，支持點分割路徑如 'model.confidence_threshold'"""
        try:
            keys = key_path.split('.')
            value = self.config
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

class PerformanceMonitor:
    """性能監控器"""
    
    def __init__(self):
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        self.detection_times = []
        
    def update_fps(self):
        """更新 FPS 計數"""
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def record_detection_time(self, detection_time: float):
        """記錄偵測時間"""
        self.detection_times.append(detection_time)
        if len(self.detection_times) > 100:  # 只保留最近100次
            self.detection_times.pop(0)
    
    def get_avg_detection_time(self) -> float:
        """獲取平均偵測時間"""
        return sum(self.detection_times) / len(self.detection_times) if self.detection_times else 0

class OptimizedMapleBot:
    """優化版 MapleStory 自動化機器人"""
    
    def __init__(self, config_path: str = "config.yaml", config: Optional[ConfigManager] = None):
        self.config = config or ConfigManager(config_path)
        self.model = None
        self.running = False
        self.paused = False
        self.hotkey_listener = None
        self.capture_failures = 0
        self.start_time = None
        self.performance_monitor = PerformanceMonitor()
        
        # 從配置載入設定
        self.monitor = self.config.get('window.default')
        self.confidence_threshold = self.config.get('model.confidence_threshold', 0.6)
        self.iou_threshold = self.config.get('model.iou_threshold', 0.45)
        self.player_templates = self._load_player_templates()
        self.player_template = self.player_templates[0] if self.player_templates else None
        self.player_model = self._load_player_model()
        self.player_facing = self.config.get('player.initial_facing', 'right')
        self.last_player: Optional[Dict] = None
        self.last_player_seen = 0.0
        self.last_turn_time = 0
        self.last_player_missing_warning = 0.0
        self.target_tracks: List[TargetTrack] = []
        self.next_target_id = 1
        self.control_state = 'idle'
        self.last_attack_time = 0.0
        self.action_delay = self.config.get('automation.action_delay', 0.3)
        self.scan_interval = self.config.get('automation.scan_interval', 0.1)
        self.cycle_count = 0
        self.last_missed_debug_time = 0.0
        self.mob_present_since: Optional[float] = None
        self.vision_lock = threading.Lock()
        self.vision_snapshot: Optional[VisionSnapshot] = None
        self.vision_thread: Optional[threading.Thread] = None
        self.control_thread: Optional[threading.Thread] = None
        self.overlay_callback = None
        self.stale_snapshot_count = 0
        self.late_vision_cycles = 0
        self.max_runtime = self.config.get('safety.max_runtime_hours', 2) * 3600
        
        # 統計數據
        self.stats = {
            'detections': 0,
            'actions_performed': 0,
            'items_collected': 0,
            'mobs_attacked': 0,
            'npcs_interacted': 0,
            'searches_performed': 0,
            'search_time_total': 0
        }
        
        # 尋找怪物相關變數
        self.last_mob_detection_time = time.time()
        self.is_searching = False
        self.search_start_time = 0
        self.original_position = None
        self.search_direction = 1  # 1 for right, -1 for left
        self.search_moves = 0
        
        # 設定 PyAutoGUI
        if self.config.get('safety.enable_failsafe', True):
            pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.05  # 減少暫停時間提升性能
        
        logger.info("OptimizedMapleBot 初始化完成")
        self._load_model()
    
    def _load_model(self):
        """載入 YOLO 模型"""
        model_path = self.config.get('model.default_path')
        if not model_path or not os.path.exists(model_path):
            logger.error(f"模型文件不存在: {model_path}")
            return False

        try:
            logger.info(f"加载模型: {model_path}")
            self.model = YOLO(model_path)
            self.model.conf = self.confidence_threshold
            self.model.iou = self.iou_threshold

            logger.info("模型加载成功")
            logger.info(f"模型类别: {self.model.names}")
            return True

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False

    def _load_player_templates(self) -> List[np.ndarray]:
        """載入角色模板庫，用於定位「我」的座標和朝向"""
        path = Path(self.config.get('player.template_path', 'assets/player_template.png'))
        if not path.exists():
            logger.warning(f"角色模板不存在: {path}")
            return []

        files = [path] if path.is_file() else sorted(
            list(path.glob('*.png')) + list(path.glob('*.jpg')) + list(path.glob('*.jpeg'))
        )
        templates = []
        for file in files:
            template = cv2.imread(str(file), cv2.IMREAD_COLOR)
            if template is None:
                logger.warning(f"角色模板加载失败: {file}")
            else:
                templates.append(template)

        if not templates:
            logger.warning("角色模板库为空")
        return templates

    def _load_player_model(self) -> Optional[YOLO]:
        """加载独立的「我」识别模型；没有该模型时继续回退模板匹配"""
        model_path = self.config.get('player.model_path', 'weights/player.pt')
        if not model_path or not os.path.exists(model_path):
            logger.info("未找到角色识别模型，角色定位将回退到模板匹配")
            return None

        try:
            model = YOLO(model_path)
            logger.info(f"✅ 角色识别模型加载成功: {model_path} | 类别: {model.names}")
            return model
        except Exception as e:
            logger.error(f"角色识别模型加载失败: {e}")
            return None

    def _detect_player_by_model(self, img: np.ndarray) -> Optional[Dict]:
        """用 player.pt 识别角色，类别本身包含 facing 信息"""
        if self.player_model is None:
            return None

        confidence = float(self.config.get('player.model_confidence', 0.30))
        fallback_confidence = float(self.config.get('player.model_fallback_confidence', 0.15))
        imgsz = int(self.config.get('player.model_imgsz', 960))
        iou = float(self.config.get('player.model_iou', 0.45))
        results = self.player_model.predict(
            img,
            conf=confidence,
            iou=iou,
            imgsz=imgsz,
            verbose=False
        )
        boxes = results[0].boxes if results else None
        if (boxes is None or len(boxes) == 0) and fallback_confidence < confidence:
            # 監控條/半遮擋場景有時會低於主閾值，用低閾值再確認一次
            results = self.player_model.predict(
                img,
                conf=fallback_confidence,
                iou=iou,
                imgsz=imgsz,
                verbose=False
            )
            boxes = results[0].boxes if results else None
        if boxes is None or len(boxes) == 0:
            return None

        best_index = int(boxes.conf.argmax().item())
        xyxy = boxes.xyxy[best_index].cpu().numpy()
        score = float(boxes.conf[best_index])
        class_id = int(boxes.cls[best_index])
        facing = self.player_model.names[class_id].replace('player_', '')
        if facing not in ('right', 'left'):
            return None

        x1, y1, x2, y2 = map(int, xyxy)
        return self._player_result({
            'score': score,
            'facing': facing,
            'location': (x1, y1),
            'size': (x2 - x1, y2 - y1)
        })

    def detect_player(self, img: np.ndarray) -> Optional[Dict]:
        """用多尺度模板匹配定位角色，同時判斷面向左或右"""
        player = self._detect_player_by_model(img)
        if player is not None:
            return player

        if not self.player_templates:
            return None

        best = None
        scales = self.config.get(
            'player.match_scales',
            [1.0, 0.95, 1.05, 0.90, 1.10, 0.85, 1.15]
        )
        for template in self.player_templates:
            candidates = (
                ('right', template),
                ('left', cv2.flip(template, 1))
            )

            for facing, candidate in candidates:
                for scale in scales:
                    scale = float(scale)
                    if scale <= 0:
                        continue
                    th, tw = candidate.shape[:2]
                    scaled_w = max(8, int(round(tw * scale)))
                    scaled_h = max(8, int(round(th * scale)))
                    if scaled_w >= img.shape[1] or scaled_h >= img.shape[0]:
                        continue

                    candidate_scaled = candidate if scale == 1.0 else cv2.resize(
                        candidate,
                        (scaled_w, scaled_h),
                        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                    )
                    result = cv2.matchTemplate(img, candidate_scaled, cv2.TM_CCOEFF_NORMED)
                    _, score, _, location = cv2.minMaxLoc(result)
                    if best is None or score > best['score']:
                        best = {
                            'score': float(score),
                            'facing': facing,
                            'location': location,
                            'size': (scaled_w, scaled_h)
                        }
                    if best['score'] >= 0.995:
                        return self._player_result(best)

        threshold = self.config.get('player.match_threshold', 0.55)
        if best is None or best['score'] < threshold:
            return None
        return self._player_result(best)

    def _player_result(self, best: Dict) -> Dict:
        """把模板匹配結果轉成主循環使用的角色座標"""
        x, y = best['location']
        width, height = best['size']
        center = (x + width // 2, y + height // 2)
        result = {
            'bbox': [x, y, x + width, y + height],
            'center': center,
            'foot_center': (center[0], y + height),
            'facing': best['facing'],
            'score': best['score']
        }
        self.player_facing = best['facing']
        return result

    def _bbox_iou(self, a: List[int], b: List[int]) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        intersection = iw * ih
        if intersection <= 0:
            return 0.0
        area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
        area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
        return intersection / float(area_a + area_b - intersection)

    def _update_target_tracks(self, detections: List[Detection]) -> None:
        """把單幀偵測結果關聯到既有目標，形成短時跟蹤"""
        now = time.time()
        target_detections = [
            d for d in detections if d.class_name in TARGET_CLASSES
        ]
        unmatched_tracks = set(range(len(self.target_tracks)))
        unmatched_detections = list(range(len(target_detections)))
        max_distance = int(self.config.get('player.target_match_distance', 90))

        # 高信賴度結果優先匹配，減少誤檢搶走既有目標
        ordered = sorted(
            unmatched_detections,
            key=lambda idx: target_detections[idx].confidence,
            reverse=True
        )
        for detection_index in ordered:
            detection = target_detections[detection_index]
            det_foot = ((detection.bbox[0] + detection.bbox[2]) // 2, detection.bbox[3])
            best_index = None
            best_cost = None

            for track_index in list(unmatched_tracks):
                track = self.target_tracks[track_index]
                if track.detection.class_name != detection.class_name:
                    continue

                iou = self._bbox_iou(track.detection.bbox, detection.bbox)
                distance = abs(track.foot_center[0] - det_foot[0]) + abs(
                    track.foot_center[1] - det_foot[1]
                )
                if iou < 0.10 and distance > max_distance:
                    continue

                cost = distance - iou * 100.0 - detection.confidence * 5.0
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_index = track_index

            if best_index is not None:
                track = self.target_tracks[best_index]
                track.detection = detection
                track.hits += 1
                track.misses = 0
                track.last_seen = now
                unmatched_tracks.remove(best_index)
                unmatched_detections.remove(detection_index)

        for detection_index in unmatched_detections:
            self.target_tracks.append(TargetTrack(
                id=self.next_target_id,
                detection=target_detections[detection_index],
                last_seen=now
            ))
            self.next_target_id += 1

        max_missed = int(self.config.get('player.max_missed_frames', 2))
        for track_index in unmatched_tracks:
            self.target_tracks[track_index].misses += 1

        self.target_tracks = [
            track for track in self.target_tracks if track.misses <= max_missed
        ]

    def get_confirmed_target_tracks(self) -> List[TargetTrack]:
        """只輸出連續出現的目標，避免單幀誤檢直接觸發動作"""
        min_hits = max(1, int(self.config.get('player.min_confirm_frames', 2)))
        return sorted(
            [t for t in self.target_tracks if t.hits >= min_hits and t.misses == 0],
            key=lambda t: t.detection.confidence,
            reverse=True
        )

    def split_target_tracks_by_player(
        self, tracks: List[TargetTrack], player: Dict
    ) -> Tuple[List[TargetTrack], List[TargetTrack]]:
        """根據角色腳底位置與朝向拆分前方/後方目標"""
        front_distance = int(self.config.get('player.front_distance', 200))
        max_dy = int(self.config.get('player.same_platform_max_dy', 45))
        player_foot = player.get('foot_center', player['center'])
        front, behind = [], []

        for track in tracks:
            dx = track.foot_center[0] - player_foot[0]
            dy = abs(track.foot_center[1] - player_foot[1])
            if dy > max_dy:
                continue

            if self.player_facing == 'right':
                is_front = 0 <= dx <= front_distance
            else:
                is_front = -front_distance <= dx <= 0

            if is_front:
                front.append(track)
            elif abs(dx) <= front_distance:
                behind.append(track)

        front.sort(key=lambda item: abs(item.foot_center[0] - player_foot[0]))
        behind.sort(key=lambda item: abs(item.foot_center[0] - player_foot[0]))
        return front, behind

    def turn_to_behind_target(self, target: Detection, player: Dict) -> bool:
        """按與目前面向相反的方向鍵轉身"""
        cooldown = float(self.config.get('player.turn_cooldown', 0.4))
        if time.time() - self.last_turn_time < cooldown:
            return False

        direction = 'left' if self.player_facing == 'right' else 'right'
        key = self.config.get(f'controls.movement_keys.{direction}', direction)
        turn_delay = float(self.config.get('player.turn_delay', 0.15))

        logger.info(
            f"目标在身后，按 {direction.upper()} 转身 "
            f"(角色面向: {self.player_facing}, 目標: {target.class_name})"
        )
        pyautogui.keyDown(key)
        time.sleep(turn_delay)
        pyautogui.keyUp(key)

        self.player_facing = 'left' if direction == 'left' else 'right'
        self.last_turn_time = time.time()
        return True

    def _attack_cooldown_active(self) -> bool:
        cooldown = float(self.config.get('automation.attack_cooldown', 0.35))
        return time.time() - self.last_attack_time < cooldown

    def _set_control_state(self, state: str) -> None:
        if self.control_state != state:
            logger.info(f"控制状态: {self.control_state} -> {state}")
            self.control_state = state

    def capture_screen(self) -> Optional[np.ndarray]:
        """優化的螢幕擷取"""
        try:
            with mss.mss() as sct:
                screenshot = sct.grab(self.monitor)
                img = np.array(screenshot)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                return img
        except Exception as e:
            logger.error(f"屏幕捕获失败: {e}")
            return None
    
    def detect_objects(self, img: np.ndarray) -> List[Detection]:
        """優化的物件偵測"""
        if self.model is None:
            return []
        
        start_time = time.time()

        try:
            detections = self._run_tiled_inference(img)
            if detections is None:
                detections = self._run_full_frame_inference(img)

            # 按優先級和距離排序
            detections = self._prioritize_detections(detections)

            # 記錄統計
            self.stats['detections'] += len(detections)
            detection_time = time.time() - start_time
            self.performance_monitor.record_detection_time(detection_time)

            return detections

        except Exception as e:
            logger.error(f"目标检测失败: {e}")
            return []

    def _run_full_frame_inference(self, img: np.ndarray) -> List[Detection]:
        """整圖推理"""
        results = self.model(img, verbose=False)
        return self._extract_detections(results)

    def _run_tiled_inference(self, img: np.ndarray) -> Optional[List[Detection]]:
        """小目標分塊推理；未啟用時返回 None"""
        cfg = self.config.get('model.tiled_inference', {})
        if not cfg.get('enabled', False):
            return None

        tile_size = int(cfg.get('size', 640))
        stride = int(cfg.get('stride', tile_size // 2))
        stride = max(1, stride)
        exclude_regions = cfg.get('exclude_regions', [])

        height, width = img.shape[:2]
        raw_detections = []
        y_starts = self._tile_starts(height, tile_size, stride)
        x_starts = self._tile_starts(width, tile_size, stride)

        for y in y_starts:
            for x in x_starts:
                crop = img[y:y + tile_size, x:x + tile_size]
                results = self.model.predict(
                    crop,
                    imgsz=tile_size,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )
                for result in results:
                    boxes = result.boxes
                    if boxes is None:
                        continue

                    for box in boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        bbox = [
                            int(xyxy[0] + x),
                            int(xyxy[1] + y),
                            int(xyxy[2] + x),
                            int(xyxy[3] + y)
                        ]

                        center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                        if self._is_excluded_detection(center, exclude_regions):
                            continue

                        raw_detections.append({
                            'bbox': bbox,
                            'confidence': confidence,
                            'class_id': class_id,
                            'center': center
                        })

        # 分塊會造成同一目標出現多個重疊框，保留信賴度最高且未被其他框大面積包含者
        raw_detections.sort(key=lambda item: item['confidence'], reverse=True)
        detections = []
        for item in raw_detections:
            if not self._is_mostly_contained(item, detections):
                detections.append(item)

        center_x, center_y = self.monitor['width'] // 2, self.monitor['height'] // 2
        return [
            Detection(
                bbox=item['bbox'],
                confidence=item['confidence'],
                class_id=item['class_id'],
                class_name=self.model.names[item['class_id']],
                center=item['center'],
                distance_from_center=float(np.sqrt(
                    (item['center'][0] - center_x) ** 2 +
                    (item['center'][1] - center_y) ** 2
                ))
            )
            for item in detections
        ]

    def _tile_starts(self, length: int, tile_size: int, stride: int) -> List[int]:
        """生成分块起点，并强制补齐右侧/底部边界"""
        length = max(1, int(length))
        tile_size = max(1, int(tile_size))
        stride = max(1, int(stride))
        if length <= tile_size:
            return [0]

        last_start = length - tile_size
        starts = list(range(0, last_start + 1, stride))
        if starts[-1] < last_start:
            starts.append(last_start)
        return starts

    def _is_excluded_detection(self, center: Tuple[int, int], regions: List) -> bool:
        """排除小地圖、任務列表等固定 UI 區域"""
        x, y = center
        for region in regions:
            try:
                rx1, ry1, rx2, ry2 = region
                if rx1 <= x <= rx2 and ry1 <= y <= ry2:
                    return True
            except (TypeError, ValueError):
                continue
        return False

    def _is_mostly_contained(self, candidate: Dict, accepted: List[Dict]) -> bool:
        """判斷分塊推理產生的低信賴度框是否只是高信賴度框的重複"""
        ax1, ay1, ax2, ay2 = candidate['bbox']
        candidate_area = max(1, (ax2 - ax1) * (ay2 - ay1))

        for item in accepted:
            if item['class_id'] != candidate['class_id']:
                continue

            bx1, by1, bx2, by2 = item['bbox']
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if intersection / candidate_area > 0.55:
                return True
        return False

    def _debug_missed_detection(self, img: np.ndarray):
        """保存空幀截圖並顯示被正式閾值過濾掉的低信賴度目標"""
        cfg = self.config.get('model.missed_detection_debug', {})
        if not cfg.get('enabled', False):
            return

        now = time.time()
        if now - self.last_missed_debug_time < float(cfg.get('interval', 2.0)):
            return
        self.last_missed_debug_time = now

        try:
            save_dir = Path(cfg.get('save_dir', 'debug_missed'))
            save_dir.mkdir(parents=True, exist_ok=True)
            results = self.model.predict(
                img,
                imgsz=640,
                conf=float(cfg.get('confidence', 0.05)),
                iou=self.iou_threshold,
                verbose=False
            )

            raw_items = []
            annotated = img.copy()
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    raw_items.append((
                        self.model.names[class_id],
                        confidence,
                        [int(v) for v in box.xyxy[0].cpu().numpy()]
                    ))
                plotted = result.plot()
                if plotted is not None and plotted.size:
                    annotated = plotted

            cv2.imwrite(str(save_dir / 'last_no_target.jpg'), img)
            cv2.imwrite(str(save_dir / 'last_no_target_debug.jpg'), annotated)
            if raw_items:
                detail = ', '.join(
                    f"{name}={confidence:.2f}@{bbox}"
                    for name, confidence, bbox in raw_items
                )
                logger.warning(f"低置信度检测结果: {detail}")
            else:
                logger.info("低置信度检测结果: 没有任何目标，已保存当前空帧")
        except Exception as e:
            logger.error(f"漏检诊断失败: {e}")

    def _nudge_position(self):
        """怪物持續停留時做一次右左微調"""
        right_key = self.config.get('controls.movement_keys.right', 'right')
        left_key = self.config.get('controls.movement_keys.left', 'left')

        pyautogui.keyDown(right_key)
        time.sleep(0.05)
        pyautogui.keyUp(right_key)
        time.sleep(0.05)
        pyautogui.keyDown(left_key)
        time.sleep(0.05)
        pyautogui.keyUp(left_key)
        logger.info("怪物持续 3 秒，执行右 -> 左微调")

    def _extract_detections(self, results) -> List[Detection]:
        """從 YOLO 結果中建立 Detection 列表"""
        center_x, center_y = self.monitor['width'] // 2, self.monitor['height'] // 2
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                if confidence <= self.confidence_threshold:
                    continue

                class_id = int(box.cls[0].cpu().numpy())
                bbox = [int(x) for x in xyxy]
                center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                distance = float(np.sqrt((center[0] - center_x) ** 2 + (center[1] - center_y) ** 2))
                detections.append(Detection(
                    bbox=bbox,
                    confidence=confidence,
                    class_id=class_id,
                    class_name=self.model.names[class_id],
                    center=center,
                    distance_from_center=distance
                ))

        return detections
    
    def _prioritize_detections(self, detections: List[Detection]) -> List[Detection]:
        """按優先級和距離排序偵測結果"""
        priority_map = {name: i for i, name in enumerate(self.config.get('automation.priority_targets', []))}
        
        def sort_key(detection):
            priority = priority_map.get(detection.class_name, 999)
            return (priority, detection.distance_from_center)
        
        return sorted(detections, key=sort_key)
    
    def perform_action(
        self,
        detection: Detection,
        distance_from_player: Optional[float] = None,
        attack_key: Optional[str] = None
    ) -> bool:
        """執行優化的遊戲動作"""
        class_name = detection.class_name
        abs_x = self.monitor['left'] + detection.center[0]
        abs_y = self.monitor['top'] + detection.center[1]
        
        # 檢查距離限制
        distance = distance_from_player if distance_from_player is not None else detection.distance_from_center
        max_distance = self.config.get(f'detection_behavior.{class_name}.max_distance', 200)
        if distance > max_distance:
            return False
        
        try:
            action_performed = False
            
            if class_name in ('mob', '木面怪人', '石面怪人'):
                # 檢查是否啟用攻擊動作
                mob_action = self.config.get(f'detection_behavior.{class_name}.action', 'attack')
                if mob_action == 'attack':
                    attack_method = self.config.get('controls.attack_method', 'click')
                    attack_key = attack_key or self.config.get('controls.attack_key', 'z')
                    if attack_method == 'key':
                        pyautogui.press(attack_key)
                    else:
                        pyautogui.moveTo(abs_x, abs_y, duration=0.1)
                        pyautogui.click()
                    logger.info(
                        f"攻击怪物 (置信度: {detection.confidence:.2f}, 按键: {attack_key})"
                    )
                    self.stats['mobs_attacked'] += 1
                    action_performed = True
                    time.sleep(self.config.get(f'detection_behavior.{class_name}.attack_delay', 0.15))
                else:
                    logger.info(f"检测到怪物 (置信度: {detection.confidence:.2f}) - 仅记录")
                
            elif class_name == 'item':
                # 只偵測物品，不執行動作
                logger.info(f"检测到物品 (置信度: {detection.confidence:.2f}) - 仅记录")
                
            elif class_name == 'npc':
                # 只偵測 NPC，不執行動作
                logger.info(f"检测到 NPC (置信度: {detection.confidence:.2f}) - 仅记录")
            
            if action_performed:
                self.stats['actions_performed'] += 1
                return True
                
        except Exception as e:
            logger.error(f"执行动作失败: {e}")
        
        return False
    
    def _should_search_for_mobs(self) -> bool:
        """檢查是否應該開始尋找怪物"""
        if not self.config.get('automation.mob_hunting.enable', True):
            return False
        
        # 如果正在搜尋中，不重複開始
        if self.is_searching:
            return False
        
        # 檢查距離上次偵測到怪物的時間
        search_delay = self.config.get('automation.mob_hunting.search_delay', 2.0)
        time_since_last_mob = time.time() - self.last_mob_detection_time
        
        return time_since_last_mob > search_delay
    
    def _start_mob_search(self):
        """開始尋找怪物"""
        if self.is_searching:
            return
        
        self.is_searching = True
        self.search_start_time = time.time()
        self.search_moves = 0
        
        # 記錄當前位置（假設角色在畫面中心）
        self.original_position = (self.monitor['width'] // 2, self.monitor['height'] // 2)
        
        logger.info("开始寻找怪物...")
    
    def _perform_mob_search(self):
        """執行尋找怪物的移動"""
        if not self.is_searching:
            return
        
        max_search_time = self.config.get('automation.mob_hunting.max_search_time', 10)
        if time.time() - self.search_start_time > max_search_time:
            self._end_mob_search()
            return
        
        search_pattern = self.config.get('automation.mob_hunting.search_pattern', 'horizontal')
        move_distance = self.config.get('automation.mob_hunting.move_distance', 100)
        
        try:
            if search_pattern == 'horizontal':
                self._horizontal_search(move_distance)
            elif search_pattern == 'vertical':
                self._vertical_search(move_distance)
            elif search_pattern == 'random':
                self._random_search(move_distance)
            
            time.sleep(0.5)  # 移動後稍作停頓
            
        except Exception as e:
            logger.error(f"搜索移动失败: {e}")
            self._end_mob_search()
    
    def _horizontal_search(self, move_distance: int):
        """水平搜尋移動"""
        move_key = self.config.get('controls.movement_keys.right' if self.search_direction > 0 else 'controls.movement_keys.left', 'right' if self.search_direction > 0 else 'left')
        
        # 按住移動鍵一段時間
        pyautogui.keyDown(move_key)
        time.sleep(0.3)
        pyautogui.keyUp(move_key)
        
        self.search_moves += 1
        
        # 每移動3次改變方向
        if self.search_moves >= 3:
            self.search_direction *= -1
            self.search_moves = 0
            logger.info(f"改变搜索方向: {'右' if self.search_direction > 0 else '左'}")
    
    def _vertical_search(self, move_distance: int):
        """垂直搜尋移動（跳躍和下降）"""
        if self.search_moves % 2 == 0:
            # 跳躍
            jump_key = self.config.get('controls.movement_keys.jump', 'x')
            pyautogui.press(jump_key)
            logger.info("跳跃搜索")
        else:
            # 向下移動
            down_key = self.config.get('controls.movement_keys.down', 'down')
            pyautogui.keyDown(down_key)
            time.sleep(0.2)
            pyautogui.keyUp(down_key)
            logger.info("向下搜索")
        
        self.search_moves += 1
    
    def _random_search(self, move_distance: int):
        """隨機搜尋移動"""
        import random
        
        movements = ['left', 'right', 'jump']
        chosen_movement = random.choice(movements)
        
        if chosen_movement == 'jump':
            jump_key = self.config.get('controls.movement_keys.jump', 'x')
            pyautogui.press(jump_key)
            logger.info("随机跳跃")
        else:
            move_key = self.config.get(f'controls.movement_keys.{chosen_movement}', chosen_movement)
            pyautogui.keyDown(move_key)
            time.sleep(0.3)
            pyautogui.keyUp(move_key)
            logger.info(f"随机移动: {chosen_movement}")
        
        self.search_moves += 1
    
    def _end_mob_search(self):
        """結束尋找怪物"""
        if not self.is_searching:
            return
        
        # 記錄搜尋統計
        search_duration = time.time() - self.search_start_time
        self.stats['searches_performed'] += 1
        self.stats['search_time_total'] += search_duration
        
        self.is_searching = False
        logger.info(f"结束怪物搜索 (耗时: {search_duration:.1f}秒)")
        
        # 如果設定要返回中心，執行返回動作
        if self.config.get('automation.mob_hunting.return_to_center', True):
            self._return_to_center()
    
    def _return_to_center(self):
        """返回到搜尋開始的位置"""
        try:
            logger.info("返回原始位置...")
            # 簡單的返回邏輯：向相反方向移動
            if self.search_direction > 0:
                # 如果最後是向右移動，現在向左移動
                move_key = self.config.get('controls.movement_keys.left', 'left')
            else:
                # 如果最後是向左移動，現在向右移動
                move_key = self.config.get('controls.movement_keys.right', 'right')
            
            pyautogui.keyDown(move_key)
            time.sleep(0.5)  # 移動時間稍長一些
            pyautogui.keyUp(move_key)
            
        except Exception as e:
            logger.error(f"返回中心失败: {e}")
    
    def _check_safety_conditions(self) -> bool:
        """檢查安全條件"""
        if self.start_time and time.time() - self.start_time > self.max_runtime:
            logger.warning("达到最大运行时间限制")
            return False
        return True
    
    def start_automation(self, show_preview: bool = False):
        """启动视觉线程与控制线程，两者通过最新检测结果快照解耦"""
        if self.model is None:
            logger.error("模型未加载，无法开始自动化")
            return
        
        self.running = True
        self.start_time = time.time()
        self.control_state = 'idle'
        self.mob_present_since = None
        self.vision_snapshot = None
        self.stale_snapshot_count = 0
        self.late_vision_cycles = 0
        self.last_attack_time = 0.0
        logger.info("开始 MapleStory Worlds 自动化")
        if self.hotkey_listener is None and not getattr(self, "hotkeys_external", False):
            self._start_hotkey_listener()
        logger.info("按 F8 暂停/恢复，F9 停止")

        self.vision_thread = threading.Thread(
            target=self._vision_worker,
            args=(show_preview,),
            name="VisionWorker",
            daemon=True
        )
        self.control_thread = threading.Thread(
            target=self._control_worker,
            name="ControlWorker",
            daemon=True
        )
        self.vision_thread.start()
        self.control_thread.start()
        try:
            while self.running and self._check_safety_conditions():
                if self.vision_thread.is_alive() and self.control_thread.is_alive():
                    time.sleep(0.1)
                    continue
                if self.running:
                    logger.error("识别或控制线程已退出，自动化停止")
                break
        except KeyboardInterrupt:
            logger.info("用户中断自动化")
        finally:
            self.running = False
            self.vision_thread.join(timeout=1.0)
            self.control_thread.join(timeout=1.0)
            if self.hotkey_listener is not None:
                self._stop_hotkey_listener()
            cv2.destroyAllWindows()
            self._log_final_statistics()
            logger.info("自动化已停止")

    def _vision_worker(self, show_preview: bool = False):
        """持续截图和推理；动作不会阻塞这里"""
        last_stats_time = time.time()
        while self.running:
            cycle_started_at = time.time()
            next_cycle_at = cycle_started_at + self.scan_interval
            try:
                if self.paused:
                    with self.vision_lock:
                        self.vision_snapshot = VisionSnapshot(
                            created_at=cycle_started_at,
                            detections=[],
                            targets=[],
                            paused=True
                        )
                    time.sleep(0.1)
                    continue

                capture_started_at = time.perf_counter()
                img = self.capture_screen()
                capture_duration = time.perf_counter() - capture_started_at
                if img is None:
                    self.capture_failures += 1
                    if self.capture_failures >= 10:
                        logger.error("屏幕捕获连续失败，请检查 macOS 屏幕录制权限后重新启动")
                        self.running = False
                        break
                    time.sleep(0.2)
                    continue
                self.capture_failures = 0
                capture_scale_x = float(img.shape[1]) / max(1, int(self.monitor["width"]))
                capture_scale_y = float(img.shape[0]) / max(1, int(self.monitor["height"]))

                detection_started_at = time.perf_counter()
                detections = self.detect_objects(img)
                inference_duration = time.perf_counter() - detection_started_at
                targets = [d for d in detections if d.class_name in TARGET_CLASSES]
                snapshot = VisionSnapshot(
                    created_at=time.time(),
                    detections=detections,
                    targets=targets,
                    capture_ms=capture_duration * 1000.0,
                    inference_ms=inference_duration * 1000.0,
                    capture_scale_x=capture_scale_x,
                    capture_scale_y=capture_scale_y
                )
                with self.vision_lock:
                    self.vision_snapshot = snapshot

                self.performance_monitor.update_fps()
                self.cycle_count += 1
                # GUI 里的预览必须走 Tk 覆盖层；cv2.imshow 在后台线程
                # 会和 Tk/Qt 的窗口栈冲突，Windows 上尤其容易报
                # "Unknown C++ exception from OpenCV code"。
                if show_preview and self.overlay_callback is None and detections:
                    preview_img = self._draw_detections(img.copy(), detections)
                    cv2.imshow('MapleStory Auto Bot - F8 暂停/恢复', preview_img)
                if show_preview and self.overlay_callback is None:
                    cv2.waitKey(1)

                if self.overlay_callback is not None:
                    try:
                        self.overlay_callback(snapshot)
                    except Exception as error:
                        logger.error(f"实时覆盖层更新失败: {error}")
                        self.overlay_callback = None

                if not targets:
                    self._debug_missed_detection(img)

                if time.time() - last_stats_time >= 30:
                    self._log_statistics()
                    last_stats_time = time.time()

                remaining = next_cycle_at - time.time()
                cycle_duration = time.time() - cycle_started_at
                if cycle_duration > self.scan_interval:
                    self.late_vision_cycles += 1
                if remaining > 0:
                    time.sleep(remaining)
            except Exception:
                logger.exception("识别线程发生未处理错误")
                time.sleep(0.2)

    def _control_worker(self):
        """按固定攻击间隔消费最新目标快照"""
        while self.running:
            try:
                if self.paused:
                    self._set_control_state('idle')
                    time.sleep(0.1)
                    continue

                with self.vision_lock:
                    snapshot = self.vision_snapshot
                now = time.time()
                # 拒绝明显过期的识别结果，避免控制线程拿着旧画面按键
                valid_for = max(0.2, self.scan_interval * 3.0)
                snapshot_is_fresh = bool(
                    snapshot and now - snapshot.created_at <= valid_for
                )
                if snapshot is not None and not snapshot_is_fresh:
                    self.stale_snapshot_count += 1
                targets = snapshot.targets if snapshot_is_fresh else []

                if targets:
                    self.last_mob_detection_time = now
                    if self.mob_present_since is None:
                        self.mob_present_since = now
                    self._set_control_state('attack')
                    cooldown = float(self.config.get('automation.attack_cooldown', 0.5))
                    if now - self.last_attack_time >= cooldown:
                        attack_key = str(self.config.get('controls.attack_key', 'alt'))
                        pyautogui.press(attack_key)
                        self.last_attack_time = time.time()
                        self.stats['actions_performed'] += 1
                        self.stats['mobs_attacked'] += 1
                        logger.info(
                            f"攻击目标 {len(targets)} 个 "
                            f"(最近识别延迟: {(time.time() - snapshot.created_at) * 1000:.0f}ms)"
                        )

                    reposition_delay = float(self.config.get('automation.stuck_reposition_after', 3.0))
                    if reposition_delay > 0 and time.time() - self.mob_present_since >= reposition_delay:
                        self._nudge_position()
                        self.mob_present_since = time.time()
                else:
                    self._set_control_state('idle')
                    self.mob_present_since = None

                time.sleep(0.02)
            except Exception as error:
                logger.error(f"控制线程发生错误: {error}")
                time.sleep(0.2)

    def get_realtime_metrics(self) -> Dict:
        """给 GUI 读取的最新流水线指标"""
        with self.vision_lock:
            snapshot = self.vision_snapshot
        now = time.time()
        return {
            'running': self.running,
            'paused': self.paused,
            'control_state': self.control_state,
            'snapshot_age_ms': (now - snapshot.created_at) * 1000.0 if snapshot else None,
            'capture_ms': snapshot.capture_ms if snapshot else None,
            'inference_ms': snapshot.inference_ms if snapshot else None,
            'target_count': len(snapshot.targets) if snapshot else 0,
            'stale_snapshot_count': self.stale_snapshot_count,
            'late_vision_cycles': self.late_vision_cycles
        }

    def _start_hotkey_listener(self):
        """启动全局快捷键监听"""
        if keyboard is None:
            logger.warning("未安装 pynput，全局快捷键不可用；仍可使用 Ctrl+C 停止")
            return

        if self.hotkey_listener is not None:
            return

        self.hotkey_listener = keyboard.Listener(on_press=self._handle_hotkey)
        self.hotkey_listener.daemon = True
        self.hotkey_listener.start()

    def _stop_hotkey_listener(self):
        """停止全局快捷键监听"""
        if self.hotkey_listener is None:
            return

        self.hotkey_listener.stop()
        self.hotkey_listener = None

    def _handle_hotkey(self, key):
        """处理全局快捷键"""
        try:
            if key == keyboard.Key.f8:
                self.paused = not self.paused
                logger.info(f"{'暂停' if self.paused else '恢复'}自动化")
            elif key == keyboard.Key.f9:
                logger.info("收到停止快捷键")
                self.running = False
        except Exception as e:
            logger.error(f"处理快捷键失败: {e}")

    def _draw_detections(self, img: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """绘制检测结果"""
        for detection in detections:
            bbox = detection.bbox
            class_name = detection.class_name
            confidence = detection.confidence
            color_map = {
                'mob': (0, 0, 255),
                'item': (0, 255, 0),
                'npc': (255, 0, 0),
                'character': (255, 255, 0),
                'environment': (128, 128, 128),
                'ui': (255, 0, 255)
            }
            color = color_map.get(class_name, (255, 255, 255))
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(img, label, (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        fps_text = f"FPS: {self.performance_monitor.current_fps}"
        cv2.putText(img, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return img

    def _log_statistics(self):
        """记录统计信息"""
        runtime = time.time() - self.start_time if self.start_time else 0
        avg_detection_time = self.performance_monitor.get_avg_detection_time()
        logger.info("运行统计:")
        logger.info(f"   运行时间: {runtime/60:.1f} 分钟")
        logger.info(f"   FPS: {self.performance_monitor.current_fps}")
        logger.info(f"   平均检测时间: {avg_detection_time*1000:.1f}ms")
        logger.info(f"   总检测次数: {self.stats['detections']}")
        logger.info(f"   执行动作: {self.stats['actions_performed']}")
        logger.info(f"   拾取物品: {self.stats['items_collected']}")
        logger.info(f"   攻击怪物: {self.stats['mobs_attacked']}")
        logger.info(f"   NPC交互: {self.stats['npcs_interacted']}")
        logger.info(f"   搜索次数: {self.stats['searches_performed']}")
        if self.stats['searches_performed'] > 0:
            avg_search_time = self.stats['search_time_total'] / self.stats['searches_performed']
            logger.info(f"   平均搜索时间: {avg_search_time:.1f}秒")

    def _log_final_statistics(self):
        """记录最终统计"""
        logger.info("最终统计报告:")
        self._log_statistics()

    def get_performance_summary(self) -> Dict:
        """获取性能摘要"""
        runtime = time.time() - self.start_time if self.start_time else 0
        avg_detection_time = self.performance_monitor.get_avg_detection_time()
        return {
            'runtime_minutes': runtime / 60,
            'current_fps': self.performance_monitor.current_fps,
            'avg_detection_time_ms': avg_detection_time * 1000,
            'total_detections': self.stats['detections'],
            'actions_performed': self.stats['actions_performed'],
            'items_collected': self.stats['items_collected'],
            'mobs_attacked': self.stats['mobs_attacked'],
            'npcs_interacted': self.stats['npcs_interacted'],
            'searches_performed': self.stats['searches_performed'],
            'avg_search_time': self.stats['search_time_total'] / max(1, self.stats['searches_performed'])
        }

    def test_detection(self):
        """测试目标检测功能"""
        if self.model is None:
            logger.error("模型未加载")
            return

        logger.info("测试目标检测功能")
        img = self.capture_screen()
        if img is None:
            logger.error("无法捕获画面")
            return

        detections = self.detect_objects(img)
        logger.info(f"检测结果: 发现 {len(detections)} 个目标")
        for i, detection in enumerate(detections, 1):
            logger.info(f"  {i}. {detection.class_name} (置信度: {detection.confidence:.2f}, 距离: {detection.distance_from_center:.0f}px)")

        if detections:
            result_img = self._draw_detections(img, detections)
            save_dir = Path("debug_missed")
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"test_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(str(save_path), result_img)
            logger.info(f"测试识别结果已保存: {save_path}")
        else:
            logger.info("未检测到任何目标")

class QueueLogHandler(logging.Handler):
    """把日誌轉發到 Tkinter 主線程顯示"""
    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        try:
            self.log_queue.put_nowait(self.format(record))
        except Exception:
            pass

class AutoControlPanel:
    """桌面控制面板"""

    def __init__(self):
        self.config = ConfigManager()
        self.models = load_available_models(verbose=False)
        self.log_queue = queue.Queue()
        self.ui_queue = queue.Queue()
        self.worker_thread: Optional[threading.Thread] = None
        self.overlay_window: Optional[tk.Toplevel] = None
        self.overlay_canvas: Optional[tk.Canvas] = None
        self.overlay_kind: Optional[str] = None
        self.native_overlay_window = None
        self.native_overlay_view = None
        self.overlay_latest_snapshot: Optional[VisionSnapshot] = None
        self.closing = False
        self.global_hotkey_monitor = None
        self.last_hotkey_name = None
        self.last_hotkey_time = 0.0
        self.measure_monitor_names = self._load_monitor_names()

        self.root = tk.Tk()
        self.root.title("MapleStory Worlds 自动化控制台")
        self.root.geometry("820x540")
        self.root.minsize(720, 480)

        self.log_handler = QueueLogHandler(self.log_queue)
        self.log_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(self.log_handler)

        self._build_style()
        self._build_layout()
        self._load_settings_to_ui()
        self._update_buttons()
        self.root.after(200, self._poll)
        self.root.after(100, self._poll_metrics)
        self.root.after(33, self._poll_overlay)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<F8>", lambda event: self._toggle_pause())
        self.root.bind("<F9>", lambda event: self._stop())

    def _build_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background="#f5f6f7")
        style.configure("TLabel", background="#f5f6f7", font=("PingFang SC", 11))
        style.configure("Header.TLabel", font=("PingFang SC", 14, "bold"))
        style.configure("Status.TLabel", font=("PingFang SC", 11, "bold"))
        style.configure("Metrics.TLabel", font=("Menlo", 10), foreground="#333333")
        style.configure("TButton", font=("PingFang SC", 11), padding=(8, 5))
        style.configure("TSpinbox", font=("PingFang SC", 11))
        style.configure("TCombobox", font=("PingFang SC", 11))

    def _build_layout(self):
        root = ttk.Frame(self.root, padding=(12, 10, 12, 8))
        root.pack(fill=tk.BOTH, expand=True)

        settings = ttk.LabelFrame(root, text="配置", padding=10)
        settings.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        ttk.Label(settings, text="怪物模型").grid(row=0, column=0, sticky="w", pady=2)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(
            settings,
            textvariable=self.model_var,
            values=list(self.models.values()),
            state="readonly"
        )
        self.model_combo.grid(row=0, column=1, columnspan=3, sticky="ew", pady=2)

        region_labels = (
            ("left", "左边"),
            ("top", "顶部"),
            ("width", "宽度"),
            ("height", "高度"),
        )
        self.region_vars = {}
        for index, (name, label) in enumerate(region_labels):
            ttk.Label(settings, text=label).grid(
                row=1 + index // 2, column=(index % 2) * 2, sticky="w", padx=(8 if index % 2 else 0, 4), pady=2
            )
            var = tk.IntVar()
            self.region_vars[name] = var
            spin = ttk.Spinbox(settings, from_=0, to=5000, textvariable=var, width=8)
            spin.grid(row=1 + index // 2, column=(index % 2) * 2 + 1, sticky="ew", pady=2)

        self.measure_button = ttk.Button(settings, text="框选监控区", command=self._measure_region)
        self.measure_button.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 2))
        ttk.Label(settings, text="显示器").grid(
            row=3, column=2, sticky="w", padx=(8, 4), pady=(6, 2)
        )
        self.measure_monitor_var = tk.StringVar(value="主显示器")
        self.measure_monitor_combo = ttk.Combobox(
            settings,
            textvariable=self.measure_monitor_var,
            values=self.measure_monitor_names,
            state="readonly",
            width=12
        )
        self.measure_monitor_combo.grid(
            row=3, column=3, sticky="ew", pady=(6, 2)
        )

        ttk.Label(settings, text="攻击键").grid(row=4, column=0, sticky="w", pady=2)
        self.attack_key_var = tk.StringVar()
        ttk.Entry(settings, textvariable=self.attack_key_var, width=10).grid(row=4, column=1, sticky="w", pady=2)

        ttk.Label(settings, text="识别间隔（秒）").grid(row=4, column=2, sticky="w", padx=(8, 4), pady=2)
        self.scan_interval_var = tk.DoubleVar()
        ttk.Entry(settings, textvariable=self.scan_interval_var, width=8).grid(row=4, column=3, sticky="w", pady=2)

        ttk.Label(settings, text="置信度").grid(row=5, column=0, sticky="w", pady=2)
        self.confidence_var = tk.DoubleVar()
        ttk.Entry(settings, textvariable=self.confidence_var, width=10).grid(row=5, column=1, sticky="w", pady=2)

        ttk.Label(settings, text="攻击间隔（秒）").grid(row=5, column=2, sticky="w", padx=(8, 4), pady=2)
        self.attack_cooldown_var = tk.DoubleVar()
        ttk.Entry(settings, textvariable=self.attack_cooldown_var, width=8).grid(row=5, column=3, sticky="w", pady=2)

        ttk.Label(settings, text="停留微调（秒）").grid(row=6, column=0, sticky="w", pady=2)
        self.reposition_delay_var = tk.DoubleVar()
        ttk.Entry(settings, textvariable=self.reposition_delay_var, width=10).grid(row=6, column=1, sticky="w", pady=2)

        ttk.Label(settings, text="漏检诊断").grid(row=6, column=2, sticky="w", padx=(8, 4), pady=2)
        self.debug_enabled_var = tk.BooleanVar()
        ttk.Checkbutton(settings, text="开启", variable=self.debug_enabled_var).grid(
            row=6, column=3, sticky="w", pady=2
        )

        ttk.Button(settings, text="保存配置", command=self._save_settings).grid(
            row=7, column=0, columnspan=4, sticky="ew", pady=(8, 0)
        )

        control = ttk.LabelFrame(root, text="控制", padding=10)
        control.grid(row=0, column=1, sticky="nsew")

        self.status_var = tk.StringVar(value="已停止")
        ttk.Label(control, textvariable=self.status_var, style="Status.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 6)
        )

        self.start_button = ttk.Button(control, text="开始自动化", command=self._start)
        self.start_button.grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=2)

        self.preview_button = ttk.Button(control, text="预览启动", command=lambda: self._start(True))
        self.preview_button.grid(row=1, column=1, sticky="ew", pady=2)

        self.pause_button = ttk.Button(control, text="暂停（F8）", command=self._toggle_pause)
        self.pause_button.grid(row=2, column=0, sticky="ew", padx=(0, 4), pady=2)

        self.stop_button = ttk.Button(control, text="停止（F9）", command=self._stop)
        self.stop_button.grid(row=2, column=1, sticky="ew", pady=2)

        ttk.Separator(control).grid(row=3, column=0, columnspan=2, sticky="ew", pady=6)
        ttk.Button(control, text="测试识别", command=self._test_detection).grid(row=4, column=0, sticky="ew", padx=(0, 4), pady=2)
        ttk.Button(control, text="漏检图片", command=self._open_debug_images).grid(row=4, column=1, sticky="ew", pady=2)
        self.overlay_enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control, text="实时画面覆盖", variable=self.overlay_enabled_var).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )
        self.metrics_var = tk.StringVar(value="尚未运行")
        ttk.Label(
            control,
            textvariable=self.metrics_var,
            style="Metrics.TLabel",
            justify=tk.LEFT,
            wraplength=240
        ).grid(row=6, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        control.columnconfigure(0, weight=1)
        control.columnconfigure(1, weight=1)

        log_frame = ttk.LabelFrame(root, text="运行日志", padding=8)
        log_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(8, 0))
        self.log_text = tk.Text(log_frame, height=10, wrap=tk.WORD, state=tk.DISABLED, font=("Menlo", 10))
        self.log_text.pack(fill=tk.BOTH, expand=True)

        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=0)
        root.rowconfigure(0, weight=0)
        root.rowconfigure(1, weight=1)

    def _load_settings_to_ui(self):
        default_path = str(self.config.get("model.default_path", ""))
        if default_path not in self.models.values() and self.models:
            default_path = next(iter(self.models.values()))
        self.model_var.set(default_path)

        monitor = self.bot_monitor()
        for name, var in self.region_vars.items():
            var.set(int(monitor[name]))
        self.attack_key_var.set(str(self.config.get("controls.attack_key", "alt")))
        self.scan_interval_var.set(float(self.config.get("automation.scan_interval", 0.5)))
        self.confidence_var.set(float(self.config.get("model.confidence_threshold", 0.30)))
        self.attack_cooldown_var.set(float(self.config.get("automation.attack_cooldown", 0.5)))
        self.reposition_delay_var.set(float(self.config.get("automation.stuck_reposition_after", 3.0)))
        self.debug_enabled_var.set(bool(self.config.get("model.missed_detection_debug.enabled", True)))
        self.overlay_enabled_var.set(bool(self.config.get("overlay.enabled", True)))

    def bot_monitor(self) -> Dict:
        return dict(self.config.get("window.default", {"left": 0, "top": 0, "width": 640, "height": 480}))

    def _save_settings(self):
        if self._is_running():
            messagebox.showwarning("提示", "请先停止自动化再修改配置")
            return
        try:
            values = {name: int(var.get()) for name, var in self.region_vars.items()}
            if values["width"] <= 0 or values["height"] <= 0:
                raise ValueError
            confidence = float(self.confidence_var.get())
            interval = float(self.scan_interval_var.get())
            attack_cooldown = float(self.attack_cooldown_var.get())
            reposition_delay = float(self.reposition_delay_var.get())
            debug_interval = float(self.config.get("model.missed_detection_debug.interval", 2.0))
            if confidence <= 0 or interval <= 0 or attack_cooldown < 0 or reposition_delay < 0:
                raise ValueError

            model_path = self.model_var.get()
            if not model_path or not Path(model_path).exists():
                messagebox.showerror("输入错误", "选择的怪物模型文件不存在")
                return

            self.config.config["model"]["default_path"] = model_path
            self.config.config["model"]["confidence_threshold"] = confidence
            self.config.config["window"]["default"] = values
            self.config.config["automation"]["scan_interval"] = interval
            self.config.config["automation"]["attack_cooldown"] = attack_cooldown
            self.config.config["automation"]["stuck_reposition_after"] = reposition_delay
            self.config.config["controls"]["attack_key"] = self.attack_key_var.get().strip() or "alt"
            self.config.config["model"]["missed_detection_debug"]["enabled"] = bool(
                self.debug_enabled_var.get()
            )
            self.config.config["model"]["missed_detection_debug"]["interval"] = debug_interval
            self.config.config["overlay"] = {
                "enabled": bool(self.overlay_enabled_var.get()),
                "click_through": True
            }

            if hasattr(self, "_bot"):
                self._bot.monitor = values.copy()
                self._bot.confidence_threshold = confidence
                self._bot.scan_interval = interval
                if self._bot.model is None or getattr(self._bot.model, "ckpt_path", None) != model_path:
                    self._bot.model = None
                    if not self._bot._load_model():
                        messagebox.showerror("模型加载失败", f"无法加载:\n{model_path}")
                        return
                else:
                    self._bot.model.conf = confidence
            self._save_config()
            logger.info(f"配置已保存并持久化，监控区: {values}")
            messagebox.showinfo("保存成功", "配置已保存到 config.yaml")
        except (ValueError, TypeError):
            messagebox.showerror("输入错误", "监控区必须是整数；间隔和置信度必须是有效数字")

    def _measure_region(self):
        if self.measure_button.instate(["disabled"]):
            return
        self.measure_button.configure(state=tk.DISABLED)
        try:
            with mss.mss() as sct:
                if not sct.monitors:
                    raise RuntimeError("无法读取显示器信息")
                monitor_index = self._selected_measure_monitor_index(sct.monitors)
                monitor = sct.monitors[monitor_index]
                if monitor["width"] <= 0 or monitor["height"] <= 0:
                    raise RuntimeError(
                        "无法读取屏幕信息。请给终端/Python 授予屏幕录制权限。"
                    )
                frame = np.array(sct.grab(monitor))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            scale_x = frame.shape[1] / monitor["width"]
            scale_y = frame.shape[0] / monitor["height"]
            self.status_var.set("拖选目标区域，按 Enter 或 Space 确认；Esc 取消")
            self.root.update_idletasks()
            x, y, width, height = cv2.selectROI(
                "框选监控区", frame, showCrosshair=True
            )
            cv2.destroyAllWindows()

            if width == 0 or height == 0:
                self.status_var.set("已取消框选")
                return

            # Retina 屏幕截图可能是系统坐标的两倍，这里换算回 mss 坐标。
            left = int(round(monitor["left"] + x / scale_x))
            top = int(round(monitor["top"] + y / scale_y))
            width = int(round(width / scale_x))
            height = int(round(height / scale_y))

            self.region_vars["left"].set(left)
            self.region_vars["top"].set(top)
            self.region_vars["width"].set(width)
            self.region_vars["height"].set(height)
            self.status_var.set(
                f"已填入区域: left={left}, top={top}, width={width}, height={height}"
            )
        except Exception as error:
            self.status_var.set(f"框选失败: {error}")
            logger.error(f"框选监控区失败: {error}")
        finally:
            self.measure_button.configure(state=tk.NORMAL)

    def _save_config(self):
        with open("config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(self.config.config, f, allow_unicode=True, sort_keys=False)

    def _is_running(self):
        return bool(self.worker_thread and self.worker_thread.is_alive())

    def _set_status(self, text):
        if threading.current_thread() is threading.main_thread():
            self.status_var.set(text)
        else:
            self.ui_queue.put(("status", text))

    def _refresh_buttons(self):
        if threading.current_thread() is threading.main_thread():
            self._update_buttons()
        else:
            self.ui_queue.put(("buttons", None))

    def _start(self, show_preview: bool = False):
        if self._is_running():
            return
        if self.bot().model is None and not self.bot()._load_model():
            messagebox.showerror("无法启动", "模型未加载")
            return
        self.bot().hotkeys_external = True
        self._start_global_hotkeys()
        self.bot().overlay_callback = self._handle_overlay_snapshot
        if self.overlay_enabled_var.get():
            self._enable_overlay()
        elif show_preview:
            logger.info("GUI 下预览通过实时画面覆盖显示；请开启“实时画面覆盖”")
        self.worker_thread = threading.Thread(
            target=self._run_worker,
            args=(show_preview,),
            daemon=True
        )
        self.worker_thread.start()

    def _run_worker(self, show_preview: bool):
        self._set_status("运行中")
        self._refresh_buttons()
        self.bot().start_automation(show_preview=show_preview)
        if not self.closing:
            self._disable_overlay()
        if not self.closing:
            self._set_status("已停止")
        self._refresh_buttons()

    def _toggle_pause(self):
        if not self._is_running():
            return
        self.bot().paused = not self.bot().paused
        self.status_var.set("已暂停" if self.bot().paused else "运行中")
        self._update_buttons()

    def _stop(self):
        self.bot().running = False
        self.bot().paused = False
        self._disable_overlay()
        self.status_var.set("正在停止...")
        self._update_buttons()

    def _enable_overlay(self):
        overlay_supported = (
            (sys.platform == "darwin" and AppKit is not None) or
            sys.platform == "win32"
        )
        if self.overlay_window is not None or not overlay_supported:
            if not overlay_supported:
                logger.warning("当前系统不支持实时覆盖层，改用预览窗口查看检测结果")
            return
        monitor = self.bot_monitor()
        width = max(1, int(monitor["width"]))
        height = max(1, int(monitor["height"]))
        if sys.platform == "darwin":
            self._enable_macos_overlay(monitor)
            return
        self.overlay_window = tk.Toplevel(self.root)
        self.overlay_window.title("实时检测覆盖层")
        self.overlay_window.overrideredirect(True)
        self.overlay_window.geometry(f"{width}x{height}+{int(monitor['left'])}+{int(monitor['top'])}")
        self.overlay_window.configure(bg="black")
        self.overlay_window.attributes("-topmost", True)
        self.overlay_window.attributes("-alpha", 1.0)
        try:
            self.overlay_window.attributes("-transparentcolor", "black")
        except tk.TclError:
            logger.warning("当前 Tk 版本不支持透明覆盖层")
        self.overlay_canvas = tk.Canvas(
            self.overlay_window,
            bg="black",
            highlightthickness=0,
            bd=0
        )
        self.overlay_canvas.pack(fill=tk.BOTH, expand=True)
        self.overlay_window.update_idletasks()
        self.overlay_latest_snapshot = None
        self.overlay_kind = "windows"
        self._set_overlay_click_through()

    def _enable_macos_overlay(self, monitor: Dict):
        if NativeOverlayView is None:
            logger.warning("macOS 原生覆盖层不可用，请确认 PyObjC 已安装")
            return
        AppKit.NSApplication.sharedApplication()
        left = int(monitor["left"])
        top = int(monitor["top"])
        width = max(1, int(monitor["width"]))
        height = max(1, int(monitor["height"]))
        primary_frame = AppKit.NSScreen.screens()[0].frame()
        cocoa_y = (
            primary_frame.origin.y + primary_frame.size.height
            - top - height
        )
        content_rect = Foundation.NSMakeRect(left, cocoa_y, width, height)
        window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            content_rect,
            AppKit.NSWindowStyleMaskBorderless,
            AppKit.NSBackingStoreBuffered,
            False
        )
        view = NativeOverlayView.alloc().initWithFrame_(content_rect)
        view.snapshot = None
        view._current_fps = 0
        window.setContentView_(view)
        window.setTitle_("实时检测覆盖层")
        window.setLevel_(AppKit.NSStatusWindowLevel)
        window.setOpaque_(False)
        window.setBackgroundColor_(AppKit.NSColor.clearColor())
        window.setHasShadow_(False)
        window.setIgnoresMouseEvents_(True)
        window.setAcceptsMouseMovedEvents_(False)
        window.setReleasedWhenClosed_(False)
        window.setCollectionBehavior_(
            AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces |
            AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
        )
        window.orderFrontRegardless()
        self.native_overlay_window = window
        self.native_overlay_view = view
        self.overlay_latest_snapshot = None
        self.overlay_kind = "macos"
        logger.info("macOS 原生透明覆盖层已启动")

    def _disable_overlay(self):
        self.bot().overlay_callback = None
        self.overlay_latest_snapshot = None
        if self.native_overlay_window is not None:
            try:
                self.native_overlay_window.orderOut_(None)
                self.native_overlay_window.close()
            except Exception:
                pass
        self.native_overlay_window = None
        self.native_overlay_view = None
        if self.overlay_window is not None:
            try:
                self.overlay_window.destroy()
            except tk.TclError:
                pass
        self.overlay_window = None
        self.overlay_canvas = None
        self.overlay_kind = None

    def _set_overlay_click_through(self):
        if self.overlay_window is None:
            return
        if sys.platform == "win32":
            self._set_windows_overlay_click_through()
            return
        try:
            app = AppKit.NSApplication.sharedApplication()
            window = app.windowWithWindowNumber_(self.overlay_window.winfo_id())
            if window is None:
                logger.warning("实时覆盖层无法设置为鼠标穿透")
                return
            window.setIgnoresMouseEvents_(True)
            window.setLevel_(AppKit.NSStatusWindowLevel)
            window.setOpaque_(False)
            window.setHasShadow_(False)
            window.setBackgroundColor_(AppKit.NSColor.clearColor())
            window.setCollectionBehavior_(
                AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
            )
        except Exception as error:
            logger.warning(f"实时覆盖层鼠标穿透设置失败: {error}")

    def _set_windows_overlay_click_through(self):
        """Win32 置顶 + 鼠标穿透；Tk 的 winfo_id 需要取父 HWND"""
        try:
            user32 = ctypes.windll.user32
            hwnd = user32.GetParent(self.overlay_window.winfo_id())
            if not hwnd:
                hwnd = self.overlay_window.winfo_id()
            GWL_EXSTYLE = -20
            WS_EX_LAYERED = 0x00080000
            WS_EX_TRANSPARENT = 0x00000020
            WS_EX_TOOLWINDOW = 0x00000080
            WS_EX_NOACTIVATE = 0x08000000
            ex_style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            user32.SetWindowLongW(
                hwnd,
                GWL_EXSTYLE,
                ex_style |
                WS_EX_LAYERED |
                WS_EX_TRANSPARENT |
                WS_EX_TOOLWINDOW |
                WS_EX_NOACTIVATE
            )
            HWND_TOPMOST = -1
            SWP_NOMOVE = 0x0002
            SWP_NOSIZE = 0x0001
            SWP_NOACTIVATE = 0x0010
            user32.SetWindowPos(
                hwnd,
                HWND_TOPMOST,
                0,
                0,
                0,
                0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE
            )
        except Exception as error:
            logger.warning(f"实时覆盖层鼠标穿透设置失败: {error}")

    def _handle_overlay_snapshot(self, snapshot: VisionSnapshot):
        # Tk 控件只能在主线程更新，视觉线程只保存最新快照
        self.overlay_latest_snapshot = snapshot

    def _draw_overlay(self):
        if self.overlay_kind == "macos":
            self._draw_macos_overlay()
            return
        if self.overlay_window is None or self.overlay_canvas is None:
            return
        monitor = self.bot_monitor()
        width = max(1, int(monitor["width"]))
        height = max(1, int(monitor["height"]))
        canvas = self.overlay_canvas
        canvas.delete("all")
        canvas.create_rectangle(
            2, 2, width - 2, height - 2,
            outline="#ffd400", width=2
        )
        snapshot = self.overlay_latest_snapshot
        if snapshot is None:
            canvas.create_text(
                12, 16, anchor="w", text="等待识别...",
                fill="#ffffff", font=("PingFang SC", 12, "bold")
            )
            return
        for detection in snapshot.detections:
            x1, y1, x2, y2 = detection.bbox
            x1 = x1 / snapshot.capture_scale_x
            y1 = y1 / snapshot.capture_scale_y
            x2 = x2 / snapshot.capture_scale_x
            y2 = y2 / snapshot.capture_scale_y
            color = "#00ff66" if detection.class_name in TARGET_CLASSES else "#8899aa"
            canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2)
            label = f"{detection.class_name} {detection.confidence:.2f}"
            canvas.create_text(
                x1 + 2, max(8, y1 - 8), anchor="w", text=label,
                fill=color, font=("PingFang SC", 10, "bold")
            )
        state = "已暂停" if snapshot.paused else ("攻击" if snapshot.targets else "待机")
        status_text = f"FPS {self.bot().performance_monitor.current_fps} | {state} | 截图 {snapshot.capture_ms:.0f}ms | 推理 {snapshot.inference_ms:.0f}ms"
        canvas.create_rectangle(
            2, height - 24, max(width - 2, 2 + len(status_text) * 7), height - 2,
            fill="black", outline=""
        )
        canvas.create_text(
            10, height - 13, anchor="w", text=status_text,
            fill="#ffffff", font=("Menlo", 10, "bold")
        )

    def _draw_macos_overlay(self):
        if self.native_overlay_view is None:
            return
        self.native_overlay_view.snapshot = self.overlay_latest_snapshot
        self.native_overlay_view._current_fps = self.bot().performance_monitor.current_fps
        self.native_overlay_view.setNeedsDisplay_(True)

    def _poll_overlay(self):
        if self.closing:
            return
        self._draw_overlay()
        self.root.after(33, self._poll_overlay)

    def _poll_metrics(self):
        if self.closing:
            return
        if not self._is_running() or not hasattr(self, "_bot"):
            self.metrics_var.set("尚未运行")
        else:
            metrics = self.bot().get_realtime_metrics()
            if metrics['paused']:
                state = "已暂停"
            elif metrics['control_state'] == 'attack':
                state = "攻击"
            else:
                state = "待机"

            def fmt_ms(value):
                return "--" if value is None else f"{value:.0f}ms"

            self.metrics_var.set(
                f"{state} | 目标 {metrics['target_count']} | 快照 {fmt_ms(metrics['snapshot_age_ms'])}\n"
                f"FPS {self.bot().performance_monitor.current_fps} | "
                f"截图 {fmt_ms(metrics['capture_ms'])} | 推理 {fmt_ms(metrics['inference_ms'])}\n"
                f"过期快照 {metrics['stale_snapshot_count']} | "
                f"迟到周期 {metrics['late_vision_cycles']}"
            )
        self.root.after(100, self._poll_metrics)

    def _test_detection(self):
        if self._is_running():
            messagebox.showwarning("提示", "请先停止自动化")
            return
        if self.bot().model is None and not self.bot()._load_model():
            return
        threading.Thread(target=self.bot().test_detection, daemon=True).start()

    def _open_debug_images(self):
        directory = Path(self.config.get("model.missed_detection_debug.save_dir", "debug_missed"))
        if not directory.exists():
            messagebox.showinfo("提示", "目前没有漏检诊断图片")
            return
        system = sys.platform
        if system == "darwin":
            os.system(f'open "{directory}"')
        elif system == "win32":
            os.system(f'start "" "{directory}"')
        else:
            os.system(f'xdg-open "{directory}"')

    def _start_global_hotkeys(self):
        if self.global_hotkey_monitor is not None:
            return

        if sys.platform == "win32":
            if keyboard is None:
                logger.warning("未安装 pynput，Windows 全局快捷键不可用；仍可使用窗口内 F8/F9")
                return

            def handle_windows_key(key):
                try:
                    if key == keyboard.Key.f8:
                        self.ui_queue.put(("hotkey", "f8"))
                    elif key == keyboard.Key.f9:
                        self.ui_queue.put(("hotkey", "f9"))
                except Exception as error:
                    logger.error(f"处理 Windows 快捷键失败: {error}")

            try:
                self.global_hotkey_monitor = keyboard.Listener(
                    on_press=handle_windows_key
                )
                self.global_hotkey_monitor.daemon = True
                self.global_hotkey_monitor.start()
                return
            except Exception as error:
                self.global_hotkey_monitor = None
                logger.warning(f"Windows 全局快捷键启动失败，仍可使用窗口内 F8/F9: {error}")
                return

        if AppKit is None:
            return
        try:
            mask = AppKit.NSEventMaskKeyDown

            def handle_key(event):
                try:
                    key_code = int(event.keyCode())
                except Exception:
                    return
                if key_code == 100:
                    self.ui_queue.put(("hotkey", "f8"))
                elif key_code == 101:
                    self.ui_queue.put(("hotkey", "f9"))

            self.global_hotkey_monitor = (
                AppKit.NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
                    mask, handle_key
                )
            )
        except Exception as error:
            logger.warning(f"全局快捷键启动失败，仍可使用窗口内 F8/F9: {error}")

    def _stop_global_hotkeys(self):
        if self.global_hotkey_monitor is None:
            return
        if sys.platform == "win32":
            try:
                self.global_hotkey_monitor.stop()
            except Exception:
                pass
        else:
            try:
                AppKit.NSEvent.removeMonitor_(self.global_hotkey_monitor)
            except Exception:
                pass
        self.global_hotkey_monitor = None

    def _handle_hotkey_name(self, name):
        now = time.time()
        if name == self.last_hotkey_name and now - self.last_hotkey_time < 0.25:
            return
        self.last_hotkey_name = name
        self.last_hotkey_time = now
        if name == "f8":
            self._toggle_pause()
        elif name == "f9":
            self._stop()

    def _load_monitor_names(self):
        names = ["主显示器", "全部桌面"]
        try:
            with mss.mss() as sct:
                for index in range(2, len(sct.monitors)):
                    names.append(f"显示器 {index}")
        except Exception:
            pass
        return names

    def _selected_measure_monitor_index(self, monitors: List[Dict]) -> int:
        name = self.measure_monitor_var.get()
        if name == "全部桌面":
            return 0
        if name.startswith("显示器 "):
            try:
                index = int(name.split()[-1])
                if 0 <= index < len(monitors):
                    return index
            except ValueError:
                pass
        return 1 if len(monitors) > 1 else 0

    def _update_buttons(self):
        running = self._is_running()
        self.start_button.configure(state=tk.DISABLED if running else tk.NORMAL)
        self.preview_button.configure(state=tk.DISABLED if running else tk.NORMAL)
        self.pause_button.configure(
            state=tk.NORMAL if running else tk.DISABLED,
            text="恢复（F8）" if self.bot().paused else "暂停（F8）"
        )
        self.stop_button.configure(state=tk.NORMAL if running else tk.DISABLED)

    def _poll(self):
        if self.closing:
            return
        while not self.log_queue.empty():
            try:
                message = self.log_queue.get_nowait()
                self.log_text.configure(state=tk.NORMAL)
                self.log_text.insert(tk.END, message + "\n")
                self.log_text.see(tk.END)
                self.log_text.configure(state=tk.DISABLED)
            except queue.Empty:
                break
        if self._is_running():
            if not self.bot().paused:
                summary = self.bot().get_performance_summary()
                self.status_var.set(
                    f"运行中 | FPS {summary['current_fps']} | 攻击 {summary['mobs_attacked']} 次"
                )
        while True:
            try:
                kind, value = self.ui_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "status":
                self.status_var.set(value)
            elif kind == "buttons":
                self._update_buttons()
            elif kind == "hotkey":
                self._handle_hotkey_name(value)
        self.root.after(300, self._poll)

    def bot(self) -> OptimizedMapleBot:
        if not hasattr(self, "_bot"):
            self._bot = OptimizedMapleBot(config=self.config)
        return self._bot

    def _on_close(self):
        self.closing = True
        if hasattr(self, "_bot"):
            self._bot.running = False
            self._bot.paused = False
            self._bot.overlay_callback = None
        self._disable_overlay()
        self._stop_global_hotkeys()
        self.root.destroy()

    def run(self):
        self.root.mainloop()
        if hasattr(self, "_bot") and self._is_running():
            self.worker_thread.join(timeout=1.5)

def load_available_models(verbose: bool = True) -> Dict[str, str]:
    """載入可用的模型文件"""
    models = {}
    weights_dir = Path("weights")
    
    if weights_dir.exists():
        for i, model_file in enumerate(sorted(weights_dir.glob("*.pt")), 1):
            size_mb = model_file.stat().st_size / (1024 * 1024)
            models[str(i)] = str(model_file)
            if verbose:
                print(f"  {i}. {model_file} ({size_mb:.1f} MB)")
    
    return models


def _select_model(models: Dict[str, str], label: str, default_key: str = '1') -> str:
    choice = input(f"\n请选择{label} (1-{len(models)}, 默认{default_key}): ").strip()
    if choice not in models:
        choice = default_key

    model_path = models[choice]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"选择的模型文件不存在: {model_path}")
    return model_path

def main():
    """主程序"""
    if sys.platform == "win32":
        # Tk/MSS/PyAutoGUI 需要同一套物理像素坐标，高缩放屏幕尤其重要
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
    if "--cli" not in sys.argv:
        AutoControlPanel().run()
        return

    print("MapleStory Worlds 自动化系统 v2.0")
    print("=" * 60)
    
    # 檢查配置文件
    if not os.path.exists("config.yaml"):
        logger.warning("配置文件不存在，将使用默认设置")
    
    # 顯示可用模型
    print("可用的模型文件:")
    models = load_available_models()
    
    if not models:
        logger.error("未找到任何模型文件")
        return
    
    # 怪物模型與角色模型各選一個；player.pt 只負責定位「我」和左右朝向
    config = ConfigManager()
    try:
        mob_default_key = next(
            (key for key, path in models.items() if Path(path).name == 'face_mobs.pt'),
            '1'
        )
        mob_path = _select_model(models, "怪物识别模型", default_key=mob_default_key)
        player_default_key = next(
            (key for key, path in models.items() if Path(path).name == 'player.pt'),
            '1'
        )
        player_path = _select_model(models, "角色识别模型", default_key=player_default_key)
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    config.config['model']['default_path'] = mob_path
    config.config['player']['model_path'] = player_path

    bot = OptimizedMapleBot(config=config)
    
    # 主選單
    while True:
        print("\n功能菜单:")
        print("1. 测试目标检测")
        print("2. 开始自动化 (有预览)")
        print("3. 开始自动化 (无预览)")
        print("4. 调整窗口设置")
        print("5. 查看配置")
        print("6. 查看统计")
        print("7. 退出")
        
        choice = input("\n请选择功能 (1-7): ").strip()
        
        if choice == '1':
            bot.test_detection()
        elif choice == '2':
            bot.start_automation(show_preview=True)
        elif choice == '3':
            bot.start_automation(show_preview=False)
        elif choice == '4':
            _adjust_window_settings(bot)
        elif choice == '5':
            _show_config(bot.config)
        elif choice == '6':
            bot._log_statistics()
        elif choice == '7':
            break
        else:
            print("无效选择")
    
    print("再见！")

def _adjust_window_settings(bot):
    """調整視窗設定"""
    print("\n当前窗口设置:")
    print(f"  左上角: ({bot.monitor['left']}, {bot.monitor['top']})")
    print(f"  大小: {bot.monitor['width']} x {bot.monitor['height']}")
    
    # 提供預設選項
    print("\n预设选项:")
    print("1. Full HD (1920x1080)")
    print("2. QHD (2560x1440)")
    print("3. 自定义设置")
    
    preset_choice = input("选择预设或自定义 (1-3): ").strip()
    
    if preset_choice == '1':
        bot.monitor = {'left': 0, 'top': 100, 'width': 1920, 'height': 980}
    elif preset_choice == '2':
        bot.monitor = {'left': 320, 'top': 180, 'width': 1280, 'height': 720}
    elif preset_choice == '3':
        try:
            bot.monitor['left'] = int(input("请输入左边位置: ") or bot.monitor['left'])
            bot.monitor['top'] = int(input("请输入顶部位置: ") or bot.monitor['top'])
            bot.monitor['width'] = int(input("请输入宽度: ") or bot.monitor['width'])
            bot.monitor['height'] = int(input("请输入高度: ") or bot.monitor['height'])
        except ValueError:
            print("输入格式错误")
            return
    
    print("窗口设置已更新")

def _show_config(config: ConfigManager):
    """顯示當前配置"""
    print("\n当前配置:")
    print(f"  怪物模型: {config.get('model.default_path')}")
    print(f"  角色模型: {config.get('player.model_path')}")
    print(f"  置信度: {config.get('model.confidence_threshold')}")
    print(f"  动作延迟: {config.get('automation.action_delay')}秒")
    print(f"  扫描间隔: {config.get('automation.scan_interval')}秒")
    print(f"  最大运行时间: {config.get('safety.max_runtime_hours')}小时")
    print(f"  拾取键: {config.get('controls.pickup_key')}")
    print(f"  交互键: {config.get('controls.interact_key')}")

if __name__ == "__main__":
    main() 
