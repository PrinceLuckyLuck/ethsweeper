"""
Ethereum Wallet Generator + Checker

Generates random private keys, computes addresses,
checks them via Bloom filter + SQLite against ~411M known addresses.

Usage:
  python generator.py              # run with default settings (30 workers)
  python generator.py --workers 16 # specify number of workers
  python generator.py --workers 16 --batch 2000  # batch size for stats
  python generator.py --benchmark              # benchmark mode (no DB required)
  python generator.py --benchmark --duration 10  # 10s benchmark
"""

import argparse
import json
import mmap
import multiprocessing
import os
import signal
import sqlite3
import sys
import tempfile
import time
from datetime import datetime

import numpy as np
import mmh3
from coincurve import PrivateKey
from Crypto.Hash import keccak



# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "eth_addresses.db")
BLOOM_PATH = os.path.join(BASE_DIR, "data", "bloom.bin")
META_PATH = os.path.join(BASE_DIR, "data", "bloom_meta.json")
FOUND_FILE = os.path.join(BASE_DIR, "found.txt")


def load_bloom_meta():
    with open(META_PATH, "r") as f:
        return json.load(f)


def bloom_check(mm, address_bytes, m, k):
    """Check if an element exists in the Bloom filter via mmap."""
    for seed in range(k):
        h = mmh3.hash(address_bytes, seed, signed=False) % m
        byte_idx = h >> 3
        bit_idx = h & 7
        if not (mm[byte_idx] & (1 << bit_idx)):
            return False
    return True


def generate_eth_address(privkey_bytes):
    """Generate an Ethereum address from 32 bytes of a private key."""
    pk = PrivateKey(privkey_bytes)
    # uncompressed public key (65 bytes: 04 + x + y), strip the 04 prefix
    pubkey = pk.public_key.format(compressed=False)[1:]
    # keccak256
    h = keccak.new(digest_bits=256)
    h.update(pubkey)
    addr_bytes = h.digest()[-20:]
    return "0x" + addr_bytes.hex()


def save_found(address, private_key_hex, eth_balance, attempt_num, db_path):
    """Save a found address to file and SQLite."""
    timestamp = datetime.now().isoformat()
    line = f"{timestamp} | {address} | {private_key_hex} | balance={eth_balance}\n"

    # Write to file
    with open(FOUND_FILE, "a") as f:
        f.write(line)

    # Write to SQLite
    try:
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT OR IGNORE INTO found (address, private_key, eth_balance, attempt_number) "
            "VALUES (?, ?, ?, ?)",
            (address, private_key_hex, eth_balance, attempt_num)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[!] SQLite write error: {e}")


def worker(worker_id, bloom_path, bloom_m, bloom_k, db_path,
           counter, found_counter, stop_event, batch_size, benchmark=False):
    """Worker process: generates keys and checks addresses."""
    # Open Bloom filter via mmap (read-only, OS shares pages)
    fd = os.open(bloom_path, os.O_RDONLY)
    mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)

    # SQLite connection (read-only) — skip in benchmark
    conn = None
    cur = None
    if not benchmark:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.execute("PRAGMA cache_size = -100000")  # ~100 MB cache per worker
        cur = conn.cursor()

    local_count = 0
    bloom_hits = 0

    try:
        while not stop_event.is_set():
            for _ in range(batch_size):
                # 1. Generate random private key
                privkey_bytes = os.urandom(32)

                # 2-3. Compute Ethereum address
                address = generate_eth_address(privkey_bytes)

                # 4. Check Bloom filter
                addr_encoded = address.encode('ascii')
                if bloom_check(mm, addr_encoded, bloom_m, bloom_k):
                    bloom_hits += 1
                    if not benchmark:
                        # 5. Confirm in SQLite
                        cur.execute("SELECT eth_balance FROM addresses WHERE address = ?",
                                    (address,))
                        row = cur.fetchone()
                        if row is not None:
                            eth_balance = row[0]
                            privkey_hex = "0x" + privkey_bytes.hex()
                            total_so_far = counter.value + local_count

                            print(f"\n{'='*60}")
                            print(f"[!!!] ADDRESS FOUND: {address}")
                            print(f"[!!!] Private key: {privkey_hex}")
                            print(f"[!!!] ETH balance: {eth_balance}")
                            print(f"[!!!] Attempt #: {total_so_far}")
                            print(f"{'='*60}\n")

                            save_found(address, privkey_hex, eth_balance, total_so_far, db_path)

                            with found_counter.get_lock():
                                found_counter.value += 1

                local_count += 1

            # Periodically update stats
            with counter.get_lock():
                counter.value += batch_size
            local_count = 0

    except KeyboardInterrupt:
        pass
    finally:
        # Send remaining count
        if local_count > 0:
            with counter.get_lock():
                counter.value += local_count

        mm.close()
        os.close(fd)
        if conn:
            conn.close()


def format_number(n):
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def main():
    parser = argparse.ArgumentParser(description="Ethereum Wallet Generator + Checker")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2),
                        help=f"Number of worker processes (default: {max(1, os.cpu_count() // 2)})")
    parser.add_argument("--batch", type=int, default=1000,
                        help="Batch size for counter updates (default: 1000)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Benchmark mode: empty bloom filter, no DB required, auto-stop")
    parser.add_argument("--duration", type=int, default=30,
                        help="Benchmark duration in seconds (default: 30)")
    args = parser.parse_args()

    if args.benchmark:
        # Benchmark mode: hardcoded bloom parameters, no files required
        bloom_m = 3_940_037_112
        bloom_k = 7
        bloom_size = (bloom_m + 7) // 8  # ~470 MB
        meta = {"n": 0, "m": bloom_m, "k": bloom_k, "fpr_actual": 0.0}

        # Create temporary random bloom file for mmap
        # Random data simulates ~50% bit density (realistic memory access pattern
        # and ~1% false positive rate, same as real filter)
        print("Creating random Bloom filter for benchmark...")
        benchmark_bloom_fd, benchmark_bloom_path = tempfile.mkstemp(suffix=".bloom")
        os.close(benchmark_bloom_fd)
        rng = np.random.default_rng(42)
        with open(benchmark_bloom_path, "wb") as f:
            # Write in 1 MB chunks to avoid huge memory allocation
            remaining = bloom_size
            while remaining > 0:
                write_size = min(remaining, 1024 * 1024)
                f.write(rng.integers(0, 256, size=write_size, dtype=np.uint8).tobytes())
                remaining -= write_size
        bloom_path = benchmark_bloom_path
        print(f"Random Bloom filter created ({bloom_size / 1024 / 1024:.1f} MB)")
    else:
        # Check required files
        for path, name in [(DB_PATH, "SQLite database"), (BLOOM_PATH, "Bloom filter"),
                           (META_PATH, "Bloom metadata")]:
            if not os.path.exists(path):
                print(f"ERROR: {name} not found: {path}")
                print("Run bloom_build.py first to build the Bloom filter.")
                sys.exit(1)

        # Load metadata
        meta = load_bloom_meta()
        bloom_m = meta["m"]
        bloom_k = meta["k"]
        bloom_path = BLOOM_PATH

    print("=" * 60)
    if args.benchmark:
        print("  Ethereum Wallet Generator - BENCHMARK MODE")
    else:
        print("  Ethereum Wallet Generator + Checker")
    print("=" * 60)
    if args.benchmark:
        print(f"  Duration:          {args.duration}s")
        print(f"  Bloom filter:      {bloom_size / 1024 / 1024:.1f} MB (random, benchmark)")
    else:
        print(f"  Addresses in DB:   {meta['n']:,}")
        print(f"  Bloom filter:      {os.path.getsize(BLOOM_PATH) / 1024 / 1024:.1f} MB "
              f"(FPR={meta['fpr_actual']:.4%})")
    print(f"  Workers:           {args.workers}")
    print(f"  Batch size:        {args.batch}")
    if not args.benchmark:
        print(f"  Found output:      {FOUND_FILE}")
    print("=" * 60)
    if args.benchmark:
        print(f"  Auto-stop after {args.duration}s")
    else:
        print("  Press Ctrl+C to stop")
    print("=" * 60)
    print()

    # Shared state
    counter = multiprocessing.Value('q', 0)       # unsigned long long
    found_counter = multiprocessing.Value('i', 0)  # int
    stop_event = multiprocessing.Event()

    # Start workers
    workers = []
    for i in range(args.workers):
        p = multiprocessing.Process(
            target=worker,
            args=(i, bloom_path, bloom_m, bloom_k, DB_PATH,
                  counter, found_counter, stop_event, args.batch, args.benchmark),
            daemon=True
        )
        p.start()
        workers.append(p)

    print(f"Started {len(workers)} worker processes")
    print()

    # Monitoring
    t0 = time.time()
    prev_count = 0
    prev_time = t0

    try:
        while True:
            time.sleep(5)
            now = time.time()
            total = counter.value
            found = found_counter.value
            elapsed = now - t0
            delta_count = total - prev_count
            delta_time = now - prev_time

            rate = delta_count / delta_time if delta_time > 0 else 0
            avg_rate = total / elapsed if elapsed > 0 else 0
            per_worker = rate / args.workers if args.workers > 0 else 0

            alive = sum(1 for p in workers if p.is_alive())

            print(f"[{elapsed:7.0f}s] Checked: {format_number(total):>10s} | "
                  f"Speed: {rate:>10,.0f} keys/sec ({per_worker:,.0f}/worker) | "
                  f"Avg: {avg_rate:>10,.0f}/sec | "
                  f"Found: {found} | Workers: {alive}/{args.workers}")

            prev_count = total
            prev_time = now

            # Check if workers are alive
            if alive == 0:
                print("\nAll workers have stopped!")
                break

            # Auto-stop for benchmark
            if args.benchmark and elapsed >= args.duration:
                print(f"\nBenchmark duration ({args.duration}s) reached.")
                stop_event.set()
                deadline = time.time() + 10
                for p in workers:
                    remaining = max(0.1, deadline - time.time())
                    p.join(timeout=remaining)
                for p in workers:
                    if p.is_alive():
                        p.terminate()
                break

    except KeyboardInterrupt:
        print("\n\nStopping...")
        stop_event.set()

        # Wait for workers to finish (max 10 sec)
        deadline = time.time() + 10
        for p in workers:
            remaining = max(0.1, deadline - time.time())
            p.join(timeout=remaining)

        # Kill stuck workers
        for p in workers:
            if p.is_alive():
                p.terminate()

    # Final stats
    total = counter.value
    found = found_counter.value
    elapsed = time.time() - t0

    print()
    print("=" * 60)
    if args.benchmark:
        print("  BENCHMARK RESULTS (CPU)")
    else:
        print("  SUMMARY")
    print("=" * 60)
    print(f"  Workers:        {args.workers}")
    print(f"  Time:           {elapsed:.1f}s")
    print(f"  Checked:        {total:,} keys")
    print(f"  Avg speed:      {total / elapsed:,.0f} keys/sec" if elapsed > 0 else "")
    if not args.benchmark:
        print(f"  Found:          {found}")
        if found > 0:
            print(f"  Results:        {FOUND_FILE}")
    print("=" * 60)

    # Cleanup benchmark temp file
    if args.benchmark:
        try:
            os.remove(benchmark_bloom_path)
        except OSError:
            pass


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
