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
import logging
import yaml
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from ultralytics import YOLO

try:
    from pynput import keyboard
except Exception:
    keyboard = None

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
                logger.warning(f"配置文件 {self.config_path} 不存在，使用默認配置")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"載入配置失敗: {e}")
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
            logger.info(f"載入模型: {model_path}")
            self.model = YOLO(model_path)
            self.model.conf = self.confidence_threshold
            self.model.iou = self.iou_threshold

            logger.info("✅ 模型載入成功!")
            logger.info(f"📊 模型類別: {self.model.names}")
            return True

        except Exception as e:
            logger.error(f"模型載入失敗: {e}")
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
                logger.warning(f"角色模板載入失敗: {file}")
            else:
                templates.append(template)

        if not templates:
            logger.warning("角色模板庫為空")
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
            f"🔄 目標在身後，按 {direction.upper()} 轉身 "
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
            logger.info(f"🎮 控制狀態: {self.control_state} -> {state}")
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
            logger.error(f"螢幕擷取失敗: {e}")
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
            logger.error(f"物件偵測失敗: {e}")
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

        for y in range(0, max(1, height - tile_size + 1), stride):
            for x in range(0, max(1, width - tile_size + 1), stride):
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
                        f"⚔️ 攻擊怪物 (信賴度: {detection.confidence:.2f}, 按鍵: {attack_key})"
                    )
                    self.stats['mobs_attacked'] += 1
                    action_performed = True
                    time.sleep(self.config.get(f'detection_behavior.{class_name}.attack_delay', 0.15))
                else:
                    logger.info(f"👁️ 偵測到怪物 (信賴度: {detection.confidence:.2f}) - 僅記錄")
                
            elif class_name == 'item':
                # 只偵測物品，不執行動作
                logger.info(f"👁️ 偵測到物品 (信賴度: {detection.confidence:.2f}) - 僅記錄")
                
            elif class_name == 'npc':
                # 只偵測 NPC，不執行動作
                logger.info(f"👁️ 偵測到 NPC (信賴度: {detection.confidence:.2f}) - 僅記錄")
            
            if action_performed:
                self.stats['actions_performed'] += 1
                return True
                
        except Exception as e:
            logger.error(f"執行動作失敗: {e}")
        
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
        
        logger.info("🔍 開始尋找怪物...")
    
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
            logger.error(f"搜尋移動失敗: {e}")
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
            logger.info(f"🔄 改變搜尋方向: {'右' if self.search_direction > 0 else '左'}")
    
    def _vertical_search(self, move_distance: int):
        """垂直搜尋移動（跳躍和下降）"""
        if self.search_moves % 2 == 0:
            # 跳躍
            jump_key = self.config.get('controls.movement_keys.jump', 'x')
            pyautogui.press(jump_key)
            logger.info("⬆️ 跳躍搜尋")
        else:
            # 向下移動
            down_key = self.config.get('controls.movement_keys.down', 'down')
            pyautogui.keyDown(down_key)
            time.sleep(0.2)
            pyautogui.keyUp(down_key)
            logger.info("⬇️ 向下搜尋")
        
        self.search_moves += 1
    
    def _random_search(self, move_distance: int):
        """隨機搜尋移動"""
        import random
        
        movements = ['left', 'right', 'jump']
        chosen_movement = random.choice(movements)
        
        if chosen_movement == 'jump':
            jump_key = self.config.get('controls.movement_keys.jump', 'x')
            pyautogui.press(jump_key)
            logger.info("🎲 隨機跳躍")
        else:
            move_key = self.config.get(f'controls.movement_keys.{chosen_movement}', chosen_movement)
            pyautogui.keyDown(move_key)
            time.sleep(0.3)
            pyautogui.keyUp(move_key)
            logger.info(f"🎲 隨機移動: {chosen_movement}")
        
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
        logger.info(f"🏁 結束怪物搜尋 (耗時: {search_duration:.1f}秒)")
        
        # 如果設定要返回中心，執行返回動作
        if self.config.get('automation.mob_hunting.return_to_center', True):
            self._return_to_center()
    
    def _return_to_center(self):
        """返回到搜尋開始的位置"""
        try:
            logger.info("🏠 返回原始位置...")
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
            logger.error(f"返回中心失敗: {e}")
    
    def _check_safety_conditions(self) -> bool:
        """檢查安全條件"""
        if self.start_time and time.time() - self.start_time > self.max_runtime:
            logger.warning("達到最大運行時間限制")
            return False
        return True
    
    def start_automation(self, show_preview: bool = False):
        """開始優化的自動化流程"""
        if self.model is None:
            logger.error("模型未載入，無法開始自動化")
            return
        
        self.running = True
        self.start_time = time.time()
        logger.info("🚀 開始 MapleStory Worlds 優化自動化")
        self._start_hotkey_listener()
        logger.info("按 F8 鍵暫停/恢復，F9 鍵停止")
        
        last_stats_time = time.time()
        
        try:
            while self.running:
                if not self._check_safety_conditions():
                    break
                
                if self.paused:
                    time.sleep(0.1)
                    continue
                
                # 擷取和偵測
                cycle_started_at = time.time()
                next_capture_at = cycle_started_at + self.scan_interval
                img = self.capture_screen()
                if img is None:
                    self.capture_failures += 1
                    if self.capture_failures >= 10:
                        logger.error("螢幕擷取連續失敗，請檢查 macOS 螢幕錄製權限後重新啟動")
                        break
                    time.sleep(1)
                    continue
                self.capture_failures = 0
                
                detections = self.detect_objects(img)
                self._update_target_tracks(detections)
                confirmed_tracks = self.get_confirmed_target_tracks()
                player = self.detect_player(img)
                if player is not None:
                    self.last_player = player
                    self.last_player_seen = cycle_started_at
                elif (
                    self.last_player is not None
                    and cycle_started_at - self.last_player_seen
                    <= float(self.config.get('player.miss_tolerance', 0.5))
                ):
                    # 模型/模板偶發漏檢時短暫沿用上一幀位置，避免誤進兜底攻擊
                    player = self.last_player

                # 舊模型仍可能輸出通用類別；新模型只輸出兩個中文怪物類別
                other_detections = [
                    d for d in detections if d.class_name not in TARGET_CLASSES
                ]
                front_tracks: List[TargetTrack] = []
                behind_tracks: List[TargetTrack] = []

                if player is not None:
                    front_tracks, behind_tracks = self.split_target_tracks_by_player(
                        confirmed_tracks, player
                    )

                # 前後一定範圍內怪物夠多時，切換為群體攻擊鍵
                attack_key = self.config.get('controls.attack_key', 'alt')
                if player is not None:
                    player_foot = player.get('foot_center', player['center'])
                    player_x = player_foot[0]
                    group_distance = int(self.config.get('player.group_attack_distance', 130))
                    group_min_targets = int(self.config.get('player.group_attack_min_targets', 2))
                    nearby_targets = [
                        t for t in front_tracks + behind_tracks
                        if abs(t.foot_center[0] - player_x) <= group_distance
                    ]
                    if len(nearby_targets) >= group_min_targets:
                        attack_key = self.config.get('player.group_attack_key', 'ctrl')

                # 檢查是否偵測到怪物，更新最後偵測時間
                mob_detected = bool(confirmed_tracks) or any(
                    d.class_name == 'mob' for d in other_detections
                )
                if mob_detected:
                    self.last_mob_detection_time = time.time()
                    # 如果正在搜尋中且偵測到怪物，停止搜尋
                    if self.is_searching:
                        self._end_mob_search()

                # 執行動作
                actions_this_cycle = 0
                for detection in other_detections:
                    if not self.running or self.paused:
                        break

                    if self.perform_action(detection):
                        actions_this_cycle += 1
                        if actions_this_cycle >= 3:
                            break
                        time.sleep(self.action_delay)

                # 控制狀態機：只有連續確認的目標才觸發轉身/攻擊
                if player is not None and front_tracks:
                    if self._attack_cooldown_active():
                        self._set_control_state('cooldown')
                    else:
                        self._set_control_state('attack')
                        target_track = front_tracks[0]
                        distance = abs(
                            target_track.foot_center[0] - player_foot[0]
                        )
                        if self.perform_action(
                            target_track.detection,
                            distance_from_player=distance,
                            attack_key=attack_key
                        ):
                            self.last_attack_time = time.time()
                            actions_this_cycle += 1
                elif player is not None and behind_tracks:
                    self._set_control_state('turn')
                    if self.turn_to_behind_target(behind_tracks[0].detection, player):
                        actions_this_cycle += 1
                elif player is None and confirmed_tracks:
                    if self._attack_cooldown_active():
                        self._set_control_state('cooldown')
                    else:
                        self._set_control_state('fallback_attack')
                        target_track = min(
                            confirmed_tracks,
                            key=lambda t: t.detection.distance_from_center
                        )
                        if time.time() - self.last_player_missing_warning >= 3:
                            logger.warning("模型與模板都未識別到角色，使用 ALT 兜底攻擊最近目標")
                            self.last_player_missing_warning = time.time()
                        if self.perform_action(
                            target_track.detection,
                            attack_key='alt'
                        ):
                            self.last_attack_time = time.time()
                            actions_this_cycle += 1
                else:
                    self._set_control_state('idle')
                
                # 如果沒有偵測到怪物且不在搜尋中，檢查是否需要開始搜尋
                if not mob_detected and self._should_search_for_mobs():
                    self._start_mob_search()
                
                # 如果正在搜尋中，執行搜尋移動
                if self.is_searching:
                    self._perform_mob_search()
                
                # 顯示預覽
                if show_preview and detections:
                    preview_img = self._draw_detections(img.copy(), detections)
                    cv2.imshow('MapleStory Auto Bot - F8 暫停/恢復', preview_img)
                
                # 更新性能監控
                self.performance_monitor.update_fps()
                
                # 定期顯示統計
                if time.time() - last_stats_time >= 30:  # 每30秒顯示一次
                    self._log_statistics()
                    last_stats_time = time.time()
                
                if show_preview:
                    cv2.waitKey(1)

                # 即使識別和動作耗時，也按目標週期排程下一次擷取
                remaining = next_capture_at - time.time()
                if remaining > 0:
                    time.sleep(remaining)
                
        except KeyboardInterrupt:
            logger.info("⏹️ 使用者中斷自動化")
        except Exception as e:
            logger.error(f"自動化過程中發生錯誤: {e}")
        finally:
            self.running = False
            self._stop_hotkey_listener()
            cv2.destroyAllWindows()
            self._log_final_statistics()
            logger.info("✅ 自動化已停止")

    def _start_hotkey_listener(self):
        """啟動全局快捷鍵監聽"""
        if keyboard is None:
            logger.warning("未安裝 pynput，全局快捷鍵不可用；仍可使用 Ctrl+C 停止")
            return

        if self.hotkey_listener is not None:
            return

        self.hotkey_listener = keyboard.Listener(on_press=self._handle_hotkey)
        self.hotkey_listener.daemon = True
        self.hotkey_listener.start()

    def _stop_hotkey_listener(self):
        """停止全局快捷鍵監聽"""
        if self.hotkey_listener is None:
            return

        self.hotkey_listener.stop()
        self.hotkey_listener = None

    def _handle_hotkey(self, key):
        """處理全局快捷鍵"""
        try:
            if key == keyboard.Key.f8:
                self.paused = not self.paused
                logger.info(f"{'⏸️ 暫停' if self.paused else '▶️ 恢復'}自動化")
            elif key == keyboard.Key.f9:
                logger.info("🛑 收到停止快捷鍵")
                self.running = False
        except Exception as e:
            logger.error(f"處理快捷鍵失敗: {e}")
    
    def _draw_detections(self, img: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """繪製偵測結果"""
        for detection in detections:
            bbox = detection.bbox
            class_name = detection.class_name
            confidence = detection.confidence
            
            # 根據類型設定顏色
            color_map = {
                'mob': (0, 0, 255),      # 紅色
                'item': (0, 255, 0),     # 綠色
                'npc': (255, 0, 0),      # 藍色
                'character': (255, 255, 0), # 青色
                'environment': (128, 128, 128), # 灰色
                'ui': (255, 0, 255)      # 洋紅色
            }
            color = color_map.get(class_name, (255, 255, 255))
            
            # 繪製邊界框
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # 繪製標籤
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(img, label, (bbox[0], bbox[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 繪製性能信息
        fps_text = f"FPS: {self.performance_monitor.current_fps}"
        cv2.putText(img, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return img
    
    def _log_statistics(self):
        """記錄統計信息"""
        runtime = time.time() - self.start_time if self.start_time else 0
        avg_detection_time = self.performance_monitor.get_avg_detection_time()
        
        logger.info("📊 運行統計:")
        logger.info(f"   運行時間: {runtime/60:.1f} 分鐘")
        logger.info(f"   FPS: {self.performance_monitor.current_fps}")
        logger.info(f"   平均偵測時間: {avg_detection_time*1000:.1f}ms")
        logger.info(f"   總偵測次數: {self.stats['detections']}")
        logger.info(f"   執行動作: {self.stats['actions_performed']}")
        logger.info(f"   撿取物品: {self.stats['items_collected']}")
        logger.info(f"   攻擊怪物: {self.stats['mobs_attacked']}")
        logger.info(f"   NPC互動: {self.stats['npcs_interacted']}")
        logger.info(f"   搜尋次數: {self.stats['searches_performed']}")
        if self.stats['searches_performed'] > 0:
            avg_search_time = self.stats['search_time_total'] / self.stats['searches_performed']
            logger.info(f"   平均搜尋時間: {avg_search_time:.1f}秒")
    
    def _log_final_statistics(self):
        """記錄最終統計"""
        logger.info("🎯 最終統計報告:")
        self._log_statistics()
    
    def get_performance_summary(self) -> Dict:
        """獲取性能摘要"""
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
        """測試偵測功能"""
        if self.model is None:
            logger.error("模型未載入")
            return
        
        logger.info("🧪 測試物件偵測功能")
        img = self.capture_screen()
        if img is None:
            logger.error("無法擷取畫面")
            return
        
        detections = self.detect_objects(img)
        logger.info(f"📊 偵測結果: 發現 {len(detections)} 個物件")
        
        for i, detection in enumerate(detections, 1):
            logger.info(f"  {i}. {detection.class_name} (信賴度: {detection.confidence:.2f}, 距離: {detection.distance_from_center:.0f}px)")
        
        if detections:
            result_img = self._draw_detections(img, detections)
            cv2.imshow('Detection Test', result_img)
            logger.info("按任意鍵關閉預覽視窗")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            logger.info("未偵測到任何物件")

def load_available_models() -> Dict[str, str]:
    """載入可用的模型文件"""
    models = {}
    weights_dir = Path("weights")
    
    if weights_dir.exists():
        for i, model_file in enumerate(sorted(weights_dir.glob("*.pt")), 1):
            size_mb = model_file.stat().st_size / (1024 * 1024)
            models[str(i)] = str(model_file)
            print(f"  {i}. {model_file} ({size_mb:.1f} MB)")
    
    return models


def _select_model(models: Dict[str, str], label: str, default_key: str = '1') -> str:
    """讓選單分別指定怪物模型與角色模型"""
    choice = input(f"\n請選擇{label} (1-{len(models)}, 預設{default_key}): ").strip()
    if choice not in models:
        choice = default_key

    model_path = models[choice]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"選擇的模型文件不存在: {model_path}")
    return model_path

def main():
    """主程序"""
    print("🍁 MapleStory Worlds 優化自動化系統 v2.0")
    print("=" * 60)
    
    # 檢查配置文件
    if not os.path.exists("config.yaml"):
        logger.warning("配置文件不存在，將使用默認設定")
    
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
        mob_path = _select_model(models, "怪物識別模型", default_key=mob_default_key)
        player_default_key = next(
            (key for key, path in models.items() if Path(path).name == 'player.pt'),
            '1'
        )
        player_path = _select_model(models, "角色識別模型", default_key=player_default_key)
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    config.config['model']['default_path'] = mob_path
    config.config['player']['model_path'] = player_path

    bot = OptimizedMapleBot(config=config)
    
    # 主選單
    while True:
        print("\n🎮 功能選單:")
        print("1. 測試物件偵測")
        print("2. 開始自動化 (有預覽)")
        print("3. 開始自動化 (無預覽)")
        print("4. 調整視窗設定")
        print("5. 查看配置")
        print("6. 查看統計")
        print("7. 退出")
        
        choice = input("\n請選擇功能 (1-7): ").strip()
        
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
            print("❌ 無效選擇")
    
    print("👋 再見！")

def _adjust_window_settings(bot):
    """調整視窗設定"""
    print(f"\n當前視窗設定:")
    print(f"  左上角: ({bot.monitor['left']}, {bot.monitor['top']})")
    print(f"  大小: {bot.monitor['width']} x {bot.monitor['height']}")
    
    # 提供預設選項
    print("\n預設選項:")
    print("1. Full HD (1920x1080)")
    print("2. QHD (2560x1440)")
    print("3. 自訂設定")
    
    preset_choice = input("選擇預設或自訂 (1-3): ").strip()
    
    if preset_choice == '1':
        bot.monitor = {'left': 0, 'top': 100, 'width': 1920, 'height': 980}
    elif preset_choice == '2':
        bot.monitor = {'left': 320, 'top': 180, 'width': 1280, 'height': 720}
    elif preset_choice == '3':
        try:
            bot.monitor['left'] = int(input("請輸入左側位置: ") or bot.monitor['left'])
            bot.monitor['top'] = int(input("請輸入頂部位置: ") or bot.monitor['top'])
            bot.monitor['width'] = int(input("請輸入寬度: ") or bot.monitor['width'])
            bot.monitor['height'] = int(input("請輸入高度: ") or bot.monitor['height'])
        except ValueError:
            print("❌ 輸入格式錯誤")
            return
    
    print("✅ 視窗設定已更新")

def _show_config(config: ConfigManager):
    """顯示當前配置"""
    print("\n⚙️ 當前配置:")
    print(f"  怪物模型: {config.get('model.default_path')}")
    print(f"  角色模型: {config.get('player.model_path')}")
    print(f"  信賴度閾值: {config.get('model.confidence_threshold')}")
    print(f"  動作延遲: {config.get('automation.action_delay')}秒")
    print(f"  掃描間隔: {config.get('automation.scan_interval')}秒")
    print(f"  最大運行時間: {config.get('safety.max_runtime_hours')}小時")
    print(f"  撿取鍵: {config.get('controls.pickup_key')}")
    print(f"  互動鍵: {config.get('controls.interact_key')}")

if __name__ == "__main__":
    main() 
