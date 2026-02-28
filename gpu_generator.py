"""
GPU Ethereum Wallet Generator + Checker (OpenCL)

Uses GPU for generating and checking Ethereum addresses.
Bloom filter is loaded into VRAM, entire pipeline runs on GPU.
Only bloom hits (~1%) are returned to CPU for SQLite confirmation.

Usage:
  python gpu_generator.py                          # default parameters
  python gpu_generator.py --global-size 262144     # number of GPU threads
  python gpu_generator.py --keys-per-thread 16     # keys per thread
  python gpu_generator.py --local-size 256         # workgroup size
  python gpu_generator.py --benchmark              # benchmark mode (no DB required)
  python gpu_generator.py --benchmark --duration 10 --no-batch  # 10s incremental benchmark
"""

import argparse
import json
import os
import signal
import sqlite3
import sys
import threading
import time
from datetime import datetime

import numpy as np

try:
    import pyopencl as cl
except ImportError:
    print("ERROR: pyopencl is not installed. Install it with:")
    print("  pip install pyopencl")
    sys.exit(1)

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KERNELS_DIR = os.path.join(BASE_DIR, "kernels")
DB_PATH = os.path.join(BASE_DIR, "data", "eth_addresses.db")
BLOOM_PATH = os.path.join(BASE_DIR, "data", "bloom.bin")
META_PATH = os.path.join(BASE_DIR, "data", "bloom_meta.json")
FOUND_FILE = os.path.join(BASE_DIR, "found.txt")

# --- Configuration ---
MAX_HITS = 65536  # Maximum bloom hits per iteration


def load_bloom_meta():
    with open(META_PATH, "r") as f:
        return json.load(f)


def save_found(address, private_key_hex, eth_balance, attempt_num):
    """Save a found address to file and SQLite."""
    timestamp = datetime.now().isoformat()
    line = f"{timestamp} | {address} | {private_key_hex} | balance={eth_balance}\n"

    with open(FOUND_FILE, "a") as f:
        f.write(line)

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT OR IGNORE INTO found (address, private_key, eth_balance, attempt_number) "
            "VALUES (?, ?, ?, ?)",
            (address, private_key_hex, eth_balance, attempt_num)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[!] SQLite write error: {e}")


def build_kernel_source():
    """Assemble kernel source code from individual .cl files."""
    # Order matters: dependencies first
    files = ["prng.cl", "murmurhash3.cl", "bloom.cl", "secp256k1.cl",
             "keccak256.cl", "eth_generator.cl"]

    source_parts = []
    for fname in files:
        path = os.path.join(KERNELS_DIR, fname)
        with open(path, "r") as f:
            source_parts.append(f"// === {fname} ===\n")
            source_parts.append(f.read())
            source_parts.append("\n\n")

    return "".join(source_parts)


def select_gpu_device():
    """Select a GPU device for OpenCL."""
    platforms = cl.get_platforms()
    for platform in platforms:
        devices = platform.get_devices(device_type=cl.device_type.GPU)
        if devices:
            # Prefer NVIDIA
            for dev in devices:
                if "nvidia" in dev.vendor.lower() or "nvidia" in platform.name.lower():
                    return platform, dev
            return platform, devices[0]

    print("ERROR: No GPU device found.")
    print("\nAvailable platforms:")
    for p in platforms:
        print(f"  {p.name}")
        for d in p.get_devices():
            print(f"    - {d.name} ({cl.device_type.to_string(d.type)})")
    sys.exit(1)


def format_number(n):
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def format_size(n_bytes):
    if n_bytes >= 1024 * 1024 * 1024:
        return f"{n_bytes / 1024 / 1024 / 1024:.1f} GB"
    if n_bytes >= 1024 * 1024:
        return f"{n_bytes / 1024 / 1024:.1f} MB"
    if n_bytes >= 1024:
        return f"{n_bytes / 1024:.1f} KB"
    return f"{n_bytes} B"


def main():
    parser = argparse.ArgumentParser(description="GPU Ethereum Wallet Generator + Checker")
    parser.add_argument("--global-size", type=int, default=262144,
                        help="Number of GPU threads (default: 262144 = 256K)")
    parser.add_argument("--local-size", type=int, default=256,
                        help="Workgroup size (default: 256)")
    parser.add_argument("--keys-per-thread", type=int, default=256,
                        help="Keys per thread per iteration (default: 256)")
    parser.add_argument("--no-incremental", action="store_true",
                        help="Use legacy kernel (full scalar mult for each key)")
    parser.add_argument("--no-batch", action="store_true",
                        help="Use incremental without batch inversion (1 mod_inv per key)")
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
    else:
        # Check required files
        for path, name in [(DB_PATH, "SQLite database"), (BLOOM_PATH, "Bloom filter"),
                           (META_PATH, "Bloom metadata")]:
            if not os.path.exists(path):
                print(f"ERROR: {name} not found: {path}")
                print("Run bloom_build.py first to build the Bloom filter.")
                sys.exit(1)

        # Load Bloom metadata
        meta = load_bloom_meta()
        bloom_m = meta["m"]
        bloom_k = meta["k"]

    # Select GPU
    platform, device = select_gpu_device()
    vram_bytes = device.global_mem_size

    print("=" * 70)
    if args.benchmark:
        print("  GPU Ethereum Wallet Generator - BENCHMARK MODE")
    else:
        print("  GPU Ethereum Wallet Generator + Checker")
    print("=" * 70)
    print(f"  GPU:               {device.name}")
    print(f"  Platform:          {platform.name}")
    print(f"  VRAM:              {format_size(vram_bytes)}")
    print(f"  Max work group:    {device.max_work_group_size}")
    print(f"  Compute units:     {device.max_compute_units}")
    if args.benchmark:
        print(f"  Duration:          {args.duration}s")
        print(f"  Bloom filter:      {format_size(bloom_size)} (random, benchmark)")
    else:
        print(f"  Addresses in DB:   {meta['n']:,}")
        print(f"  Bloom filter:      {os.path.getsize(BLOOM_PATH) / 1024 / 1024:.1f} MB "
              f"(FPR={meta['fpr_actual']:.4%})")
    print(f"  Global size:       {args.global_size:,}")
    print(f"  Local size:        {args.local_size}")
    print(f"  Keys/thread:       {args.keys_per_thread}")
    if args.no_incremental:
        mode = "full scalar mult"
    elif args.no_batch:
        mode = "incremental (P+G)"
    else:
        mode = "batch inversion (4x)"
        # Round keys_per_thread down to multiple of 4
        if args.keys_per_thread % 4 != 0:
            args.keys_per_thread = (args.keys_per_thread // 4) * 4
            if args.keys_per_thread < 4:
                args.keys_per_thread = 4
    print(f"  Mode:              {mode}")
    keys_per_iter = args.global_size * args.keys_per_thread
    print(f"  Keys/iteration:    {keys_per_iter:,}")
    print("=" * 70)

    # VRAM budget
    if not args.benchmark:
        bloom_size = os.path.getsize(BLOOM_PATH)
    seeds_size = args.global_size * 4 * 8  # 4 ulongs per thread
    hits_pk_size = MAX_HITS * 32
    hits_addr_size = MAX_HITS * 42
    hit_count_size = 4
    total_vram = bloom_size + seeds_size + hits_pk_size + hits_addr_size + hit_count_size

    print(f"\n  VRAM budget:")
    print(f"    Bloom filter:    {format_size(bloom_size)}")
    print(f"    Seeds:           {format_size(seeds_size)}")
    print(f"    Hit buffers:     {format_size(hits_pk_size + hits_addr_size)}")
    print(f"    Total:           {format_size(total_vram)} / {format_size(vram_bytes)}")

    if total_vram > vram_bytes * 0.9:
        print(f"\nWARNING: Not enough VRAM! Reduce --global-size")
        sys.exit(1)

    print("=" * 70)
    print("  Press Ctrl+C to stop")
    print("=" * 70)
    print()

    # Create OpenCL context
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    # Compile kernel
    print("Compiling OpenCL kernel...")
    kernel_source = build_kernel_source()

    # Remove #include from eth_generator.cl (already concatenated)
    kernel_source = kernel_source.replace('#include "prng.cl"', '')
    kernel_source = kernel_source.replace('#include "murmurhash3.cl"', '')
    kernel_source = kernel_source.replace('#include "bloom.cl"', '')
    kernel_source = kernel_source.replace('#include "secp256k1.cl"', '')
    kernel_source = kernel_source.replace('#include "keccak256.cl"', '')

    try:
        program = cl.Program(ctx, kernel_source).build(
            options=[f"-I{KERNELS_DIR}"]
        )
    except cl.RuntimeError as e:
        print(f"Kernel compilation ERROR:")
        print(e)
        if hasattr(program, 'get_build_info'):
            log = program.get_build_info(device, cl.program_build_info.LOG)
            print(f"Build log:\n{log}")
        sys.exit(1)

    print("Kernel compiled successfully!")

    # Load Bloom filter into VRAM
    if args.benchmark:
        print("Creating random Bloom filter in VRAM (benchmark)...")
        # Random data simulates ~50% bit density (realistic memory access pattern)
        # Real bloom also has ~50% bits set. This gives realistic bloom check cost
        # and ~1% false positive rate (same as real filter).
        rng = np.random.default_rng(42)
        bloom_data = rng.integers(0, 256, size=bloom_size, dtype=np.uint8)
        bloom_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=bloom_data)
        del bloom_data
        print(f"Random Bloom filter created ({format_size(bloom_size)})")
    else:
        print("Loading Bloom filter into VRAM...")
        with open(BLOOM_PATH, "rb") as f:
            bloom_data = f.read()
        bloom_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=bloom_data)
        del bloom_data  # Free RAM
        print(f"Bloom filter loaded into VRAM ({format_size(bloom_size)})")

    # Create buffers
    seeds_np = np.empty(args.global_size * 4, dtype=np.uint64)
    seeds_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY, size=seeds_np.nbytes)

    hit_count_np = np.zeros(1, dtype=np.uint32)
    hit_count_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=4)

    hit_privkeys_np = np.empty(MAX_HITS * 32, dtype=np.uint8)
    hit_privkeys_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=hit_privkeys_np.nbytes)

    hit_addresses_np = np.empty(MAX_HITS * 42, dtype=np.uint8)
    hit_addresses_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=hit_addresses_np.nbytes)

    # Select kernel
    if args.no_incremental:
        kernel = program.generate_and_check
    elif args.no_batch:
        kernel = program.generate_and_check_incremental
    else:
        kernel = program.generate_and_check_batch

    # --- Async hit processing (background thread for SQLite verification) ---
    # GPU launches next kernel while CPU processes bloom hits from previous iteration.
    # SQLite is I/O-bound, so GIL is released during queries.
    BATCH_SQL_SIZE = 500  # addresses per batch SQL query

    total_found = 0
    found_lock = threading.Lock()
    hit_thread = None

    def process_hits_batch(pk_data, addr_data, actual_hits, total_keys_snapshot):
        """Process bloom hits in background: batch SQLite verification."""
        nonlocal total_found
        # Open own SQLite connection (thread-safe)
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        conn.execute("PRAGMA cache_size = -200000")
        cur = conn.cursor()

        # Parse addresses and privkeys
        hits = []
        for i in range(actual_hits):
            addr_str = bytes(addr_data[i * 42:(i + 1) * 42]).decode('ascii', errors='replace')
            hits.append(addr_str)

        # Batch SQL: check BATCH_SQL_SIZE addresses at once
        found_addrs = {}
        for batch_start in range(0, len(hits), BATCH_SQL_SIZE):
            batch = hits[batch_start:batch_start + BATCH_SQL_SIZE]
            placeholders = ",".join("?" * len(batch))
            cur.execute(
                f"SELECT address, eth_balance FROM addresses WHERE address IN ({placeholders})",
                batch
            )
            for row in cur.fetchall():
                found_addrs[row[0]] = row[1]

        conn.close()

        # Process confirmed matches
        if found_addrs:
            for i in range(actual_hits):
                addr_str = hits[i]
                if addr_str in found_addrs:
                    eth_balance = found_addrs[addr_str]
                    privkey_bytes_le = bytes(pk_data[i * 32:(i + 1) * 32])
                    privkey_bytes = privkey_bytes_le[::-1]
                    privkey_hex = "0x" + privkey_bytes.hex()

                    print(f"\n{'=' * 60}")
                    print(f"[!!!] ADDRESS FOUND: {addr_str}")
                    print(f"[!!!] Private key: {privkey_hex}")
                    print(f"[!!!] ETH balance: {eth_balance}")
                    print(f"[!!!] Attempt #: {total_keys_snapshot}")
                    print(f"{'=' * 60}\n")

                    save_found(addr_str, privkey_hex, eth_balance, total_keys_snapshot)
                    with found_lock:
                        total_found += 1

    # Stats
    total_keys = 0
    total_bloom_hits = 0
    t0 = time.time()
    prev_keys = 0
    prev_time = t0

    stop = False

    def signal_handler(sig, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, signal_handler)

    if args.benchmark:
        print(f"Starting benchmark on {device.name} ({args.duration}s)...\n")
    else:
        print(f"Starting generation on {device.name}...\n")

    iteration = 0
    while not stop:
        # Auto-stop for benchmark
        if args.benchmark and (time.time() - t0) >= args.duration:
            break
        iteration += 1

        # 1. Generate random seeds on CPU
        seeds_bytes = os.urandom(args.global_size * 32)
        seeds_np = np.frombuffer(seeds_bytes, dtype=np.uint64)

        # 2. Upload seeds to VRAM
        cl.enqueue_copy(queue, seeds_buf, seeds_np)

        # 3. Reset hit counter
        hit_count_np[0] = 0
        cl.enqueue_copy(queue, hit_count_buf, hit_count_np)

        # 4. Launch kernel
        kernel(
            queue,
            (args.global_size,),
            (args.local_size,),
            seeds_buf,
            bloom_buf,
            np.uint64(bloom_m),
            np.int32(bloom_k),
            np.int32(args.keys_per_thread),
            hit_count_buf,
            hit_privkeys_buf,
            hit_addresses_buf,
            np.uint32(MAX_HITS)
        )

        # 5. Read results
        cl.enqueue_copy(queue, hit_count_np, hit_count_buf)
        queue.finish()

        n_hits = int(hit_count_np[0])
        keys_this_iter = args.global_size * args.keys_per_thread
        total_keys += keys_this_iter
        total_bloom_hits += n_hits

        # 6. Process bloom hits: read buffers, launch async SQLite verification
        if n_hits > 0 and not args.benchmark:
            actual_hits = min(n_hits, MAX_HITS)

            # Read hit buffers from GPU (PCIe transfer — must finish before next kernel)
            pk_read = np.empty(MAX_HITS * 32, dtype=np.uint8)
            addr_read = np.empty(MAX_HITS * 42, dtype=np.uint8)
            cl.enqueue_copy(queue, pk_read, hit_privkeys_buf)
            cl.enqueue_copy(queue, addr_read, hit_addresses_buf)
            queue.finish()

            # Wait for previous hit processing thread to finish
            if hit_thread is not None:
                hit_thread.join()

            # Copy data for background thread (GPU buffers will be reused)
            pk_copy = bytes(pk_read)
            addr_copy = bytes(addr_read)

            # Launch async SQLite verification — GPU proceeds to next kernel
            hit_thread = threading.Thread(
                target=process_hits_batch,
                args=(pk_copy, addr_copy, actual_hits, total_keys),
                daemon=True
            )
            hit_thread.start()

        # 7. Stats (every ~5 seconds or on first iteration)
        now = time.time()
        if now - prev_time >= 5.0 or iteration == 1:
            elapsed = now - t0
            delta_keys = total_keys - prev_keys
            delta_time = now - prev_time

            rate = delta_keys / delta_time if delta_time > 0 else 0
            avg_rate = total_keys / elapsed if elapsed > 0 else 0
            bloom_hit_pct = (total_bloom_hits / total_keys * 100) if total_keys > 0 else 0

            with found_lock:
                found = total_found

            print(f"[{elapsed:7.0f}s] Checked: {format_number(total_keys):>10s} | "
                  f"Speed: {format_number(rate):>8s} keys/sec | "
                  f"Avg: {format_number(avg_rate):>8s}/sec | "
                  f"Bloom hits: {total_bloom_hits} ({bloom_hit_pct:.3f}%) | "
                  f"Found: {found}")

            prev_keys = total_keys
            prev_time = now

    # Shutdown — wait for last hit processing
    if hit_thread is not None:
        hit_thread.join()

    elapsed = time.time() - t0
    print()
    print("=" * 70)
    if args.benchmark:
        print("  BENCHMARK RESULTS (GPU)")
    else:
        print("  SUMMARY (GPU)")
    print("=" * 70)
    print(f"  GPU:            {device.name}")
    if args.benchmark:
        print(f"  Mode:           {mode}")
    print(f"  Time:           {elapsed:.1f}s")
    print(f"  Checked:        {total_keys:,} keys")
    if elapsed > 0:
        print(f"  Avg speed:      {total_keys / elapsed:,.0f} keys/sec")
    if not args.benchmark:
        print(f"  Bloom hits:     {total_bloom_hits:,} ({total_bloom_hits / total_keys * 100:.3f}%)"
              if total_keys > 0 else "")
        print(f"  Found:          {total_found}")
        if total_found > 0:
            print(f"  Results:        {FOUND_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
