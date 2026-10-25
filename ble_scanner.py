"""BLE second-factor scanner for the face-recognition access-control system.
Updated for dynamic, arbitrary BLE tag enrollment.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import logging
import os
import subprocess
import sys
import threading
import time

log = logging.getLogger(__name__)

BLE_SCAN_INTERVAL = float(os.environ.get("BLE_SCAN_INTERVAL", "1.0"))
BLE_SCAN_DURATION = float(os.environ.get("BLE_SCAN_DURATION", "2.0"))
BLE_RSSI_FLOOR    = int(os.environ.get("BLE_RSSI_FLOOR", "-78"))
BLE_RESULT_TTL    = float(os.environ.get("BLE_RESULT_TTL", "3.5"))
BLE_ENABLED       = os.environ.get("BLE_DISABLED", "").strip() not in {"1", "true", "yes"}

MFG_COMPANY_ID = 0xFFFF


@dataclass
class ScanEntry:
    address:   str
    rssi:     int
    uuids:     list[str]
    mfr_data:  dict[int, bytes]
    timestamp: float = field(default_factory=time.monotonic)


def verify_tag(user_ble_id: str | None, scan_results: dict[str, ScanEntry]) -> bool:
    """Return True if the specific dynamic BLE identifier stored for this user is visible."""
    if not user_ble_id:
        return True  # Face-only user

    target_id = user_ble_id.strip().lower()
    now = time.monotonic()

    for entry in scan_results.values():
        if now - entry.timestamp > BLE_RESULT_TTL:
            continue
        if entry.rssi < BLE_RSSI_FLOOR:
            continue

        # Check manufacturer data payload (e.g., string identifier flashed on ESP32)
        payload = entry.mfr_data.get(MFG_COMPANY_ID)
        if payload:
            mfr_text = payload.decode("utf-8", errors="ignore").strip().lower()
            if mfr_text == target_id:
                return True

        # Check if the target ID matches any service UUID string fragment
        for u in entry.uuids:
            if target_id in u:
                return True

    return False


class BLEScanner:
    def __init__(self) -> None:
        self.enabled        = BLE_ENABLED
        self.latest_results: dict[str, ScanEntry] = {}
        self.last_scan_time: float | None = None
        self._stop_event    = threading.Event()
        self._lock          = threading.Lock()
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen[str] | None = None

    def start(self) -> None:
        if not self.enabled:
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="ble-scanner", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except OSError:
                pass

    def get_active_tag_ids(self) -> set[str]:
        if not self.enabled:
            return set()
        now = time.monotonic()
        active = set()
        with self._lock:
            entries = list(self.latest_results.values())
        for entry in entries:
            if now - entry.timestamp > BLE_RESULT_TTL:
                continue
            if entry.rssi < BLE_RSSI_FLOOR:
                continue
            payload = entry.mfr_data.get(MFG_COMPANY_ID)
            if payload:
                tag = payload.decode("utf-8", errors="ignore").strip().lower()
                if tag:
                    active.add(tag)
            for u in entry.uuids:
                active.add(u.strip().lower())
        return active

    def _run(self) -> None:
        script_path = os.path.abspath(__file__)
        while not self._stop_event.is_set():
            try:
                self._proc = subprocess.Popen(
                    [sys.executable, "-u", script_path, "--worker"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    bufsize=1,
                )
                assert self._proc.stdout is not None
                for line in self._proc.stdout:
                    if self._stop_event.is_set():
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw_items = json.loads(line)
                    except ValueError:
                        continue

                    now = time.monotonic()
                    fresh: dict[str, ScanEntry] = {}
                    with self._lock:
                        prev_results = dict(self.latest_results)
                    for addr, item in raw_items.items():
                        raw_rssi = int(item["rssi"])
                        prev_entry = prev_results.get(addr)
                        if prev_entry is not None:
                            smoothed_rssi = int(round(0.5 * raw_rssi + 0.5 * prev_entry.rssi))
                        else:
                            smoothed_rssi = raw_rssi
                        fresh[addr] = ScanEntry(
                            address=addr,
                            rssi=smoothed_rssi,
                            uuids=list(item["uuids"]),
                            mfr_data={
                                int(cid): bytes.fromhex(hex_val)
                                for cid, hex_val in item.get("mfr_data", {}).items()
                            },
                            timestamp=now,
                        )
                    with self._lock:
                        self.latest_results = fresh
                        self.last_scan_time = now
            except Exception:
                log.exception("BLE scanner supervisor error.")
            finally:
                if self._proc and self._proc.poll() is None:
                    try:
                        self._proc.terminate()
                    except OSError:
                        pass

            deadline = time.monotonic() + BLE_SCAN_INTERVAL
            while not self._stop_event.is_set() and time.monotonic() < deadline:
                time.sleep(0.25)


async def _worker_async_loop() -> None:
    from bleak import BleakScanner

    await asyncio.sleep(BLE_SCAN_INTERVAL)
    while True:
        payload_out: dict[str, dict] = {}
        try:
            devices = await BleakScanner.discover(timeout=BLE_SCAN_DURATION, return_adv=True)
            for address, (_device, adv) in devices.items():
                mfr_raw = adv.manufacturer_data or {}
                # Capture any device broadcasting our mfr company ID or any service UUIDs
                if MFG_COMPANY_ID in mfr_raw or adv.service_uuids or (adv.local_name and "TAG" in adv.local_name):
                    uuids = [str(u).lower() for u in (adv.service_uuids or [])]
                    addr_key = address.upper()
                    payload_out[addr_key] = {
                        "rssi": adv.rssi if adv.rssi is not None else -999,
                        "uuids": uuids,
                        "mfr_data": {
                            str(int(cid)): bytes(data).hex()
                            for cid, data in mfr_raw.items()
                        },
                    }
        except Exception as exc:
            sys.stderr.write(f"BLE worker scan error: {exc}\n")

        print(json.dumps(payload_out), flush=True)
        await asyncio.sleep(BLE_SCAN_INTERVAL)


def _run_worker() -> None:
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.ole32.CoInitializeEx(None, 0)
        except Exception:
            pass
    asyncio.run(_worker_async_loop())


if __name__ == "__main__":
    if "--worker" in sys.argv:
        _run_worker()