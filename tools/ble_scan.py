"""Scan for WAVELETECH wristband BLE services and characteristics."""

import asyncio
import sys
from bleak import BleakScanner, BleakClient


async def scan(duration: float = 10.0):
    print(f"[BLE] Scanning for {duration}s — make sure USB dongle is UNPLUGGED and wristband is ON...\n")

    devices = await BleakScanner.discover(timeout=duration, return_adv=True)

    if not devices:
        print("[BLE] No devices found.")
        return

    print(f"[BLE] Found {len(devices)} device(s):\n")
    print(f"{'NAME':<30} {'ADDRESS':<20} {'RSSI':>5}  UUIDS")
    print("-" * 100)

    candidates = []
    for addr, (dev, adv) in sorted(devices.items(), key=lambda x: x[1][1].rssi or -999, reverse=True):
        name = dev.name or adv.local_name or "(unknown)"
        rssi = adv.rssi or 0
        uuids = adv.service_uuids or []
        uuid_str = ", ".join(uuids) if uuids else "—"
        print(f"{name:<30} {addr:<20} {rssi:>5}  {uuid_str}")

        name_lower = (name or "").lower()
        if any(kw in name_lower for kw in ["wave", "emg", "oymotion", "gforce", "myo", "flex", "wristband", "armband"]):
            candidates.append((dev, adv))

    if candidates:
        print(f"\n[BLE] Likely wristband candidates: {len(candidates)}")
        for dev, adv in candidates:
            print(f"  -> {dev.name} ({dev.address})")
    else:
        print("\n[BLE] No obvious wristband name found. Listing all devices with service UUIDs for manual inspection.")
        for addr, (dev, adv) in devices.items():
            if adv.service_uuids:
                print(f"  -> {dev.name or '(unknown)'} ({addr}) — {adv.service_uuids}")

    return devices


async def inspect(address: str):
    print(f"\n[BLE] Connecting to {address}...")
    async with BleakClient(address) as client:
        print(f"[BLE] Connected: {client.is_connected}")
        print(f"[BLE] MTU: {client.mtu_size}")
        print(f"\n{'SERVICE UUID':<40} {'CHAR UUID':<40} {'PROPERTIES'}")
        print("=" * 120)

        for service in client.services:
            print(f"\n[Service] {service.uuid}  — {service.description or ''}")
            for char in service.characteristics:
                props = ", ".join(char.properties)
                print(f"  [Char] {char.uuid:<40} {props}")
                for desc in char.descriptors:
                    val = await client.read_gatt_descriptor(desc.handle)
                    print(f"    [Desc] {desc.uuid}: {val}")

                if "notify" in char.properties or "indicate" in char.properties:
                    print(f"  *** DATA CANDIDATE: {char.uuid} (supports notify) ***")


async def sniff(address: str, char_uuid: str, seconds: float = 10.0):
    """Subscribe to a characteristic and dump raw bytes."""
    frames = []

    def callback(sender, data: bytearray):
        frames.append(bytes(data))
        hex_str = data.hex()
        header_match = "D2D2D2" in hex_str.upper()
        flag = ""
        if header_match:
            flag = " <-- HEADER MATCH!"
        print(f"  [{len(frames):>4}] len={len(data):>3}  {hex_str[:80]}{'...' if len(hex_str)>80 else ''}{flag}")

    print(f"\n[BLE] Sniffing {char_uuid} on {address} for {seconds}s...")
    async with BleakClient(address) as client:
        await client.start_notify(char_uuid, callback)
        await asyncio.sleep(seconds)
        await client.stop_notify(char_uuid)

    print(f"\n[BLE] Captured {len(frames)} frames")
    if frames:
        sizes = set(len(f) for f in frames)
        print(f"[BLE] Frame sizes: {sizes}")
        has_header = any(b"\xD2\xD2\xD2" in f for f in frames)
        print(f"[BLE] Contains 0xD2D2D2 header: {has_header}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "scan":
        asyncio.run(scan())
    elif sys.argv[1] == "inspect" and len(sys.argv) >= 3:
        asyncio.run(inspect(sys.argv[2]))
    elif sys.argv[1] == "sniff" and len(sys.argv) >= 4:
        dur = float(sys.argv[4]) if len(sys.argv) >= 5 else 10.0
        asyncio.run(sniff(sys.argv[2], sys.argv[3], dur))
    else:
        print("Usage:")
        print("  python ble_scan.py scan                          # Scan for devices")
        print("  python ble_scan.py inspect <ADDRESS>             # List services/chars")
        print("  python ble_scan.py sniff <ADDRESS> <CHAR_UUID> [seconds]  # Capture data")
