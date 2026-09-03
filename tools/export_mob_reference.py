#!/usr/bin/env python3
"""Export monster icons from the mxdc.dvg.cn list API."""

import argparse
import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path


def safe_filename(value: str) -> str:
    return re.sub(r'[\\/:*?"<>|\s]+', "_", value).strip("._") or "unnamed"


def fetch_json(url: str, timeout: float = 30.0) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def download(url: str, path: Path, timeout: float = 30.0) -> bool:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                return False
            path.write_bytes(response.read())
        return True
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="https://mxdc.dvg.cn")
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("datasets/reference/mob_page1"))
    parser.add_argument("--delay", type=float, default=0.15)
    args = parser.parse_args()

    query = urllib.parse.urlencode({
        "page": args.page,
        "pageSize": args.page_size,
        "sortKey": "mobid",
        "sortDir": "asc",
    })
    api_url = f"{args.base_url}/api/mob-list.php?{query}"
    payload = fetch_json(api_url)

    if not payload.get("ok"):
        print(f"API returned failure: {payload}", file=sys.stderr)
        return 1

    items = payload.get("items", [])
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []

    for item in items:
        mob_id = item.get("mobid")
        name = safe_filename(str(item.get("mobname", "unnamed")))
        icon_path = item.get("icon")
        if not mob_id or not icon_path:
            continue

        icon_url = urllib.parse.urljoin(args.base_url, icon_path)
        image_name = f"{mob_id}_{name}{Path(icon_path).suffix or '.png'}"
        image_path = args.output / image_name
        downloaded = download(icon_url, image_path)

        rows.append({
            "mobid": mob_id,
            "mobname": item.get("mobname", ""),
            "level": item.get("level", ""),
            "category_label": item.get("categoryLabel", ""),
            "species_label": item.get("speciesLabel", ""),
            "boss": item.get("boss", ""),
            "icon_url": icon_url,
            "file": image_name if downloaded else "",
        })
        status = "ok" if downloaded else "failed"
        print(f"[{status}] {image_name}")
        time.sleep(args.delay)

    csv_path = args.output / "mobs.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [
            "mobid", "mobname", "level", "category_label", "species_label", "boss", "icon_url", "file"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} records to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
