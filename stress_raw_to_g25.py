"""
Concurrent stress test for POST /raw-to-g25.

Usage:
  pip install httpx
  python stress_raw_to_g25.py --url http://localhost:8000 --file path/to/raw.txt --users 10

Note: main.py defaults MAX_CONCURRENT_CONVERSIONS = 5 (env MAX_CONCURRENT_CONVERSIONS to override).
Use --users 10 (or higher) so some requests queue and you exercise the semaphore under burst load.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import os
import sys
import time

import httpx


async def one_request(
    client: httpx.AsyncClient,
    url: str,
    raw_bytes: bytes,
    vendor: str,
    idx: int,
) -> tuple[int, int, float, str | None]:
    files = {"file": ("sample.txt", io.BytesIO(raw_bytes), "text/plain")}
    data = {"vendor": vendor, "compressed": "false"}
    t0 = time.perf_counter()
    try:
        r = await client.post(url, files=files, data=data, timeout=600.0)
        dt = time.perf_counter() - t0
        err = None if r.status_code == 200 else r.text[:200]
        return idx, r.status_code, dt, err
    except Exception as e:
        dt = time.perf_counter() - t0
        return idx, -1, dt, str(e)


async def main_async(args: argparse.Namespace) -> None:
    with open(args.file, "rb") as f:
        raw_bytes = f.read()

    url = args.url.rstrip("/") + "/raw-to-g25"
    n = args.users

    print(f"POST {url}")
    print(f"Concurrent clients: {n}, file size: {len(raw_bytes) / 1024 / 1024:.2f} MB")
    print("Firing all requests at once...\n")

    t_wall0 = time.perf_counter()
    async with httpx.AsyncClient() as client:
        tasks = [
            one_request(client, url, raw_bytes, args.vendor, i) for i in range(n)
        ]
        results = await asyncio.gather(*tasks)
    wall = time.perf_counter() - t_wall0

    results.sort(key=lambda x: x[0])
    ok = sum(1 for _, code, _, _ in results if code == 200)
    for idx, code, dt, err in results:
        line = f"  #{idx}: status={code} time={dt:.1f}s"
        if err:
            line += f" err={err!r}"
        print(line)

    print(f"\nWall-clock for full burst: {wall:.1f}s")
    print(f"Success: {ok}/{n}")
    times = [dt for _, code, dt, _ in results if code == 200]
    if times:
        print(f"Per-request times (successful): min={min(times):.1f}s max={max(times):.1f}s")


def main() -> None:
    p = argparse.ArgumentParser(description="Stress test /raw-to-g25")
    p.add_argument("--url", default="http://localhost:8000", help="Base URL (no trailing path)")
    p.add_argument(
        "--file",
        required=True,
        metavar="PATH",
        help="Path to your real raw DNA export (e.g. 23andme .txt), not a placeholder name",
    )
    p.add_argument("--users", type=int, default=5, help="Number of concurrent POSTs")
    p.add_argument("--vendor", default="23andme", help="vendor form field")
    args = p.parse_args()
    path = os.path.abspath(os.path.expanduser(args.file))
    if not os.path.isfile(path):
        print(f"File not found: {path}", file=sys.stderr)
        print("Use the full path to your 23andme/Ancestry/FTDNA raw file.", file=sys.stderr)
        sys.exit(1)
    args.file = path
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
