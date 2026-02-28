"""
Тесты корректности GPU реализации.

Проверяет:
1. MurmurHash3: GPU vs Python mmh3
2. Keccak256: GPU vs pycryptodome
3. secp256k1: GPU pubkey vs coincurve
4. Full pipeline: GPU address == CPU address
5. Bloom filter: GPU bloom check vs Python bloom check

Использование:
  python test_gpu_correctness.py
  python test_gpu_correctness.py --num-keys 100    # количество ключей для теста
  python test_gpu_correctness.py --test mmh3       # только тест mmh3
"""

import argparse
import json
import os
import struct
import sys
import time

import numpy as np

try:
    import pyopencl as cl
except ImportError:
    print("ОШИБКА: pyopencl не установлен")
    sys.exit(1)

import mmh3
from coincurve import PrivateKey
from Crypto.Hash import keccak

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KERNELS_DIR = os.path.join(BASE_DIR, "kernels")
BLOOM_PATH = os.path.join(BASE_DIR, "data", "bloom.bin")
META_PATH = os.path.join(BASE_DIR, "data", "bloom_meta.json")


def select_gpu():
    platforms = cl.get_platforms()
    for platform in platforms:
        devices = platform.get_devices(device_type=cl.device_type.GPU)
        if devices:
            for dev in devices:
                if "nvidia" in dev.vendor.lower():
                    return platform, dev
            return platform, devices[0]
    # Fallback to CPU
    for platform in platforms:
        devices = platform.get_devices(device_type=cl.device_type.CPU)
        if devices:
            print("ВНИМАНИЕ: GPU не найден, используем CPU OpenCL")
            return platform, devices[0]
    print("ОШИБКА: Нет доступных OpenCL устройств")
    sys.exit(1)


def build_source(*filenames):
    parts = []
    for fname in filenames:
        path = os.path.join(KERNELS_DIR, fname)
        with open(path, "r") as f:
            parts.append(f.read())
        parts.append("\n\n")
    return "".join(parts)


def cpu_eth_address(privkey_bytes_le):
    """CPU reference: privkey bytes (LE) -> Ethereum address string."""
    pk = PrivateKey(privkey_bytes_le[::-1])  # coincurve expects BE
    pubkey = pk.public_key.format(compressed=False)[1:]  # 64 bytes
    h = keccak.new(digest_bits=256)
    h.update(pubkey)
    addr_bytes = h.digest()[-20:]
    return "0x" + addr_bytes.hex()


def cpu_pubkey(privkey_bytes_le):
    """CPU reference: privkey bytes (LE) -> uncompressed pubkey (64 bytes)."""
    pk = PrivateKey(privkey_bytes_le[::-1])  # coincurve expects BE
    return pk.public_key.format(compressed=False)[1:]


def cpu_keccak256(data):
    """CPU reference: keccak256 hash."""
    h = keccak.new(digest_bits=256)
    h.update(data)
    return h.digest()


# ============================================================
# Test 1: MurmurHash3
# ============================================================
def test_mmh3(ctx, queue, device, num_tests=1000):
    print(f"\n{'='*60}")
    print(f"  Test 1: MurmurHash3 (GPU vs Python mmh3)")
    print(f"{'='*60}")

    source = build_source("murmurhash3.cl")
    source += """
__kernel void test_mmh3(
    __global const uchar *data,
    __global const uint *lengths,
    __global const uint *seeds,
    __global uint *results,
    const int count
) {
    int gid = get_global_id(0);
    if (gid >= count) return;

    // Find data offset
    uint offset = 0;
    for (int i = 0; i < gid; i++) offset += lengths[i];

    uchar local_data[64];
    uint len = lengths[gid];
    for (uint i = 0; i < len && i < 64; i++)
        local_data[i] = data[offset + i];

    results[gid] = murmurhash3_32(local_data, (int)len, seeds[gid]);
}
"""
    program = cl.Program(ctx, source).build()

    # Generate test data
    test_strings = []
    test_seeds = []
    for i in range(num_tests):
        # Mix of address-like strings and random data
        if i % 2 == 0:
            s = f"0x{os.urandom(20).hex()}".encode('ascii')
        else:
            s = os.urandom(np.random.randint(1, 50))
        test_strings.append(s)
        test_seeds.append(np.random.randint(0, 100))

    # Flatten data
    all_data = b"".join(test_strings)
    lengths = np.array([len(s) for s in test_strings], dtype=np.uint32)
    seeds = np.array(test_seeds, dtype=np.uint32)

    # CPU reference
    cpu_results = []
    for s, seed in zip(test_strings, test_seeds):
        h = mmh3.hash(s, seed, signed=False)
        cpu_results.append(h)
    cpu_results = np.array(cpu_results, dtype=np.uint32)

    # GPU
    data_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                         hostbuf=np.frombuffer(all_data, dtype=np.uint8))
    lengths_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                            hostbuf=lengths)
    seeds_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                          hostbuf=seeds)
    results_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=num_tests * 4)

    program.test_mmh3(queue, (num_tests,), None,
                      data_buf, lengths_buf, seeds_buf, results_buf, np.int32(num_tests))

    gpu_results = np.empty(num_tests, dtype=np.uint32)
    cl.enqueue_copy(queue, gpu_results, results_buf)
    queue.finish()

    # Compare
    matches = np.sum(gpu_results == cpu_results)
    mismatches = num_tests - matches

    if mismatches > 0:
        print(f"  FAIL: {mismatches}/{num_tests} mismatches!")
        for i in range(min(5, num_tests)):
            if gpu_results[i] != cpu_results[i]:
                print(f"    [{i}] seed={test_seeds[i]} data={test_strings[i][:30]}...")
                print(f"         CPU: {cpu_results[i]:#010x}  GPU: {gpu_results[i]:#010x}")
        return False
    else:
        print(f"  OK: {num_tests}/{num_tests} hashes match")
        return True


# ============================================================
# Test 2: Keccak256
# ============================================================
def test_keccak(ctx, queue, device, num_tests=100):
    print(f"\n{'='*60}")
    print(f"  Test 2: Keccak-256 (GPU vs pycryptodome)")
    print(f"{'='*60}")

    source = build_source("keccak256.cl")
    source += """
__kernel void test_keccak(
    __global const uchar *inputs,   // num_tests * 64 bytes
    __global uchar *outputs,        // num_tests * 32 bytes
    const int count
) {
    int gid = get_global_id(0);
    if (gid >= count) return;

    uchar input[64];
    for (int i = 0; i < 64; i++)
        input[i] = inputs[gid * 64 + i];

    uchar output[32];
    keccak256_64(input, output);

    for (int i = 0; i < 32; i++)
        outputs[gid * 32 + i] = output[i];
}
"""
    program = cl.Program(ctx, source).build()

    # Generate random 64-byte inputs
    inputs = np.frombuffer(os.urandom(num_tests * 64), dtype=np.uint8)

    # CPU reference
    cpu_outputs = np.empty(num_tests * 32, dtype=np.uint8)
    for i in range(num_tests):
        data = bytes(inputs[i * 64:(i + 1) * 64])
        h = cpu_keccak256(data)
        cpu_outputs[i * 32:(i + 1) * 32] = np.frombuffer(h, dtype=np.uint8)

    # GPU
    inputs_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=inputs)
    outputs_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=num_tests * 32)

    program.test_keccak(queue, (num_tests,), None,
                        inputs_buf, outputs_buf, np.int32(num_tests))

    gpu_outputs = np.empty(num_tests * 32, dtype=np.uint8)
    cl.enqueue_copy(queue, gpu_outputs, outputs_buf)
    queue.finish()

    matches = 0
    for i in range(num_tests):
        cpu_h = cpu_outputs[i * 32:(i + 1) * 32]
        gpu_h = gpu_outputs[i * 32:(i + 1) * 32]
        if np.array_equal(cpu_h, gpu_h):
            matches += 1
        elif matches == i:  # Print first few mismatches
            print(f"  MISMATCH [{i}]:")
            print(f"    CPU: {bytes(cpu_h).hex()}")
            print(f"    GPU: {bytes(gpu_h).hex()}")

    if matches == num_tests:
        print(f"  OK: {num_tests}/{num_tests} hashes match")
        return True
    else:
        print(f"  FAIL: {matches}/{num_tests} match ({num_tests - matches} mismatches)")
        return False


# ============================================================
# Test 3: secp256k1 (pubkey generation)
# ============================================================
def test_secp256k1(ctx, queue, device, num_tests=100):
    print(f"\n{'='*60}")
    print(f"  Test 3: secp256k1 pubkey (GPU vs coincurve)")
    print(f"{'='*60}")

    source = build_source("secp256k1.cl")
    source += """
__kernel void test_secp256k1(
    __global const ulong *privkeys,  // num_tests * 4 ulongs
    __global uchar *pubkeys,         // num_tests * 64 bytes
    const int count
) {
    int gid = get_global_id(0);
    if (gid >= count) return;

    ulong pk[4];
    pk[0] = privkeys[gid * 4 + 0];
    pk[1] = privkeys[gid * 4 + 1];
    pk[2] = privkeys[gid * 4 + 2];
    pk[3] = privkeys[gid * 4 + 3];

    uchar pubkey[64];
    privkey_to_pubkey(pk, pubkey);

    for (int i = 0; i < 64; i++)
        pubkeys[gid * 64 + i] = pubkey[i];
}
"""

    print("  Компиляция secp256k1 kernel (может занять время)...")
    t_compile = time.time()
    program = cl.Program(ctx, source).build()
    print(f"  Компиляция: {time.time() - t_compile:.1f}с")

    # Generate random private keys (ensure valid: 1 <= k < n)
    privkeys_bytes = []
    privkeys_np = np.empty(num_tests * 4, dtype=np.uint64)
    for i in range(num_tests):
        pk_bytes = os.urandom(32)
        # Ensure non-zero
        while pk_bytes == b'\x00' * 32:
            pk_bytes = os.urandom(32)
        privkeys_bytes.append(pk_bytes)
        vals = np.frombuffer(pk_bytes, dtype=np.uint64)
        privkeys_np[i * 4:(i + 1) * 4] = vals

    # CPU reference
    cpu_pubkeys = []
    for pk_bytes in privkeys_bytes:
        pub = cpu_pubkey(pk_bytes)
        cpu_pubkeys.append(pub)

    # GPU
    privkeys_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=privkeys_np)
    pubkeys_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=num_tests * 64)

    print(f"  Запуск kernel для {num_tests} ключей...")
    t_run = time.time()
    program.test_secp256k1(queue, (num_tests,), None,
                           privkeys_buf, pubkeys_buf, np.int32(num_tests))

    gpu_pubkeys = np.empty(num_tests * 64, dtype=np.uint8)
    cl.enqueue_copy(queue, gpu_pubkeys, pubkeys_buf)
    queue.finish()
    print(f"  Kernel выполнен: {time.time() - t_run:.1f}с")

    matches = 0
    first_mismatch = None
    for i in range(num_tests):
        cpu_pub = cpu_pubkeys[i]
        gpu_pub = bytes(gpu_pubkeys[i * 64:(i + 1) * 64])
        if cpu_pub == gpu_pub:
            matches += 1
        elif first_mismatch is None:
            first_mismatch = i
            print(f"  MISMATCH [{i}]:")
            print(f"    privkey: {privkeys_bytes[i].hex()}")
            print(f"    CPU X:   {cpu_pub[:32].hex()}")
            print(f"    GPU X:   {gpu_pub[:32].hex()}")
            print(f"    CPU Y:   {cpu_pub[32:].hex()}")
            print(f"    GPU Y:   {gpu_pub[32:].hex()}")

    if matches == num_tests:
        print(f"  OK: {num_tests}/{num_tests} pubkeys match")
        return True
    else:
        print(f"  FAIL: {matches}/{num_tests} match ({num_tests - matches} mismatches)")
        return False


# ============================================================
# Test 4: Full pipeline (address generation)
# ============================================================
def test_full_pipeline(ctx, queue, device, num_tests=100):
    print(f"\n{'='*60}")
    print(f"  Test 4: Full pipeline - address generation (GPU vs CPU)")
    print(f"{'='*60}")

    source = build_source("prng.cl", "murmurhash3.cl", "secp256k1.cl", "keccak256.cl")
    source += """
__kernel void test_full_pipeline(
    __global const ulong *privkeys,   // num_tests * 4 ulongs
    __global uchar *addresses,        // num_tests * 42 bytes
    const int count
) {
    int gid = get_global_id(0);
    if (gid >= count) return;

    ulong pk[4];
    pk[0] = privkeys[gid * 4 + 0];
    pk[1] = privkeys[gid * 4 + 1];
    pk[2] = privkeys[gid * 4 + 2];
    pk[3] = privkeys[gid * 4 + 3];

    uchar pubkey[64];
    privkey_to_pubkey(pk, pubkey);

    uchar hash[32];
    keccak256_64(pubkey, hash);

    uchar addr[42];
    hash_to_eth_address(hash, addr);

    for (int i = 0; i < 42; i++)
        addresses[gid * 42 + i] = addr[i];
}
"""

    print("  Компиляция full pipeline kernel...")
    program = cl.Program(ctx, source).build()
    print("  OK")

    # Generate private keys
    privkeys_bytes = []
    privkeys_np = np.empty(num_tests * 4, dtype=np.uint64)
    for i in range(num_tests):
        pk_bytes = os.urandom(32)
        while pk_bytes == b'\x00' * 32:
            pk_bytes = os.urandom(32)
        privkeys_bytes.append(pk_bytes)
        vals = np.frombuffer(pk_bytes, dtype=np.uint64)
        privkeys_np[i * 4:(i + 1) * 4] = vals

    # CPU reference
    cpu_addresses = [cpu_eth_address(pk) for pk in privkeys_bytes]

    # GPU
    privkeys_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=privkeys_np)
    addresses_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=num_tests * 42)

    program.test_full_pipeline(queue, (num_tests,), None,
                               privkeys_buf, addresses_buf, np.int32(num_tests))

    gpu_addresses_raw = np.empty(num_tests * 42, dtype=np.uint8)
    cl.enqueue_copy(queue, gpu_addresses_raw, addresses_buf)
    queue.finish()

    matches = 0
    for i in range(num_tests):
        gpu_addr = bytes(gpu_addresses_raw[i * 42:(i + 1) * 42]).decode('ascii', errors='replace')
        cpu_addr = cpu_addresses[i]
        if gpu_addr == cpu_addr:
            matches += 1
        elif matches == i and i < 5:
            print(f"  MISMATCH [{i}]:")
            print(f"    privkey: {privkeys_bytes[i].hex()}")
            print(f"    CPU: {cpu_addr}")
            print(f"    GPU: {gpu_addr}")

    if matches == num_tests:
        print(f"  OK: {num_tests}/{num_tests} addresses match")
        return True
    else:
        print(f"  FAIL: {matches}/{num_tests} match ({num_tests - matches} mismatches)")
        return False


# ============================================================
# Test 5: Bloom filter
# ============================================================
def test_bloom(ctx, queue, device, num_tests=100):
    print(f"\n{'='*60}")
    print(f"  Test 5: Bloom filter check (GPU vs CPU)")
    print(f"{'='*60}")

    if not os.path.exists(BLOOM_PATH) or not os.path.exists(META_PATH):
        print("  SKIP: bloom.bin или bloom_meta.json не найдены")
        return True

    meta = json.load(open(META_PATH))
    bloom_m = meta["m"]
    bloom_k = meta["k"]

    source = build_source("murmurhash3.cl", "bloom.cl")
    source += """
__kernel void test_bloom(
    __global const uchar *bloom_data,
    __global const uchar *addresses,    // num_tests * 42 bytes
    __global int *results,
    const ulong bloom_m,
    const int bloom_k,
    const int count
) {
    int gid = get_global_id(0);
    if (gid >= count) return;

    uchar addr[42];
    for (int i = 0; i < 42; i++)
        addr[i] = addresses[gid * 42 + i];

    results[gid] = bloom_check(bloom_data, addr, 42, bloom_m, bloom_k);
}
"""
    program = cl.Program(ctx, source).build()

    # Load bloom filter
    with open(BLOOM_PATH, "rb") as f:
        bloom_bytes = f.read()
    bloom_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                          hostbuf=np.frombuffer(bloom_bytes, dtype=np.uint8))

    # Generate test addresses
    import mmap as mmap_mod
    fd = os.open(BLOOM_PATH, os.O_RDONLY)
    mm = mmap_mod.mmap(fd, 0, access=mmap_mod.ACCESS_READ)

    test_addresses = []
    for i in range(num_tests):
        addr = f"0x{os.urandom(20).hex()}"
        test_addresses.append(addr.encode('ascii'))

    # CPU reference using same bloom check logic
    cpu_results = []
    for addr_bytes in test_addresses:
        hit = True
        for seed in range(bloom_k):
            h = mmh3.hash(addr_bytes, seed, signed=False) % bloom_m
            byte_idx = h >> 3
            bit_idx = h & 7
            if not (mm[byte_idx] & (1 << bit_idx)):
                hit = False
                break
        cpu_results.append(1 if hit else 0)

    mm.close()
    os.close(fd)

    cpu_results = np.array(cpu_results, dtype=np.int32)

    # GPU
    all_addrs = b"".join(test_addresses)
    addrs_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                          hostbuf=np.frombuffer(all_addrs, dtype=np.uint8))
    results_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=num_tests * 4)

    program.test_bloom(queue, (num_tests,), None,
                       bloom_buf, addrs_buf, results_buf,
                       np.uint64(bloom_m), np.int32(bloom_k), np.int32(num_tests))

    gpu_results = np.empty(num_tests, dtype=np.int32)
    cl.enqueue_copy(queue, gpu_results, results_buf)
    queue.finish()

    matches = np.sum(gpu_results == cpu_results)
    bloom_positives = np.sum(gpu_results == 1)

    if matches == num_tests:
        print(f"  OK: {num_tests}/{num_tests} bloom checks match")
        print(f"  Bloom positives: {bloom_positives}/{num_tests} "
              f"({bloom_positives / num_tests * 100:.1f}%)")
        return True
    else:
        print(f"  FAIL: {matches}/{num_tests} match")
        return False


# ============================================================
# Test 6: Incremental key generation
# ============================================================
def test_incremental(ctx, queue, device, num_tests=100):
    """Test incremental kernel: k, k+1, k+2, ... produce correct addresses."""
    print(f"\n{'='*60}")
    print(f"  Test 6: Incremental key generation (GPU vs CPU)")
    print(f"{'='*60}")

    # Number of sequential keys per base key
    keys_per_thread = 16

    source = build_source("prng.cl", "murmurhash3.cl", "bloom.cl",
                          "secp256k1.cl", "keccak256.cl")
    source += """
__kernel void test_incremental(
    __global const ulong *privkeys,     // num_tests * 4 ulongs (base keys)
    __global uchar *addresses,          // num_tests * keys_per_thread * 42 bytes
    __global uchar *out_privkeys,       // num_tests * keys_per_thread * 32 bytes
    const int keys_per_thread,
    const int count
) {
    int gid = get_global_id(0);
    if (gid >= count) return;

    // Load base private key
    u256 k;
    k[0] = (uint)(privkeys[gid * 4 + 0] & 0xFFFFFFFF);
    k[1] = (uint)(privkeys[gid * 4 + 0] >> 32);
    k[2] = (uint)(privkeys[gid * 4 + 1] & 0xFFFFFFFF);
    k[3] = (uint)(privkeys[gid * 4 + 1] >> 32);
    k[4] = (uint)(privkeys[gid * 4 + 2] & 0xFFFFFFFF);
    k[5] = (uint)(privkeys[gid * 4 + 2] >> 32);
    k[6] = (uint)(privkeys[gid * 4 + 3] & 0xFFFFFFFF);
    k[7] = (uint)(privkeys[gid * 4 + 3] >> 32);

    // Full scalar mult for base key
    u256 px, py, pz;
    scalar_mult_G(px, py, pz, k);

    for (int iter = 0; iter < keys_per_thread; iter++) {
        int slot = gid * keys_per_thread + iter;

        // Convert current k to output privkey bytes (LE)
        ulong cur_pk[4];
        u256_to_privkey(k, cur_pk);
        __global uchar *pk_out = &out_privkeys[slot * 32];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 8; j++) {
                pk_out[i * 8 + j] = (uchar)(cur_pk[i] >> (j * 8));
            }
        }

        // Jacobian -> affine -> pubkey -> keccak -> address
        uchar pubkey[64];
        jacobian_to_pubkey(px, py, pz, pubkey);

        uchar hash[32];
        keccak256_64(pubkey, hash);

        uchar addr[42];
        hash_to_eth_address(hash, addr);

        __global uchar *addr_out = &addresses[slot * 42];
        for (int i = 0; i < 42; i++)
            addr_out[i] = addr[i];

        // P = P + G
        u256 tx, ty, tz;
        u256_copy(tx, px); u256_copy(ty, py); u256_copy(tz, pz);
        point_add_mixed(px, py, pz, tx, ty, tz, SECP256K1_GX, SECP256K1_GY);

        // k++
        u256_inc(k);
    }
}
"""

    print("  Компиляция incremental kernel...")
    t_compile = time.time()
    program = cl.Program(ctx, source).build()
    print(f"  Компиляция: {time.time() - t_compile:.1f}с")

    # Generate random base private keys
    privkeys_bytes = []
    privkeys_np = np.empty(num_tests * 4, dtype=np.uint64)
    for i in range(num_tests):
        pk_bytes = os.urandom(32)
        while pk_bytes == b'\x00' * 32:
            pk_bytes = os.urandom(32)
        privkeys_bytes.append(pk_bytes)
        vals = np.frombuffer(pk_bytes, dtype=np.uint64)
        privkeys_np[i * 4:(i + 1) * 4] = vals

    # CPU reference: compute address for k, k+1, ..., k+keys_per_thread-1
    total_addrs = num_tests * keys_per_thread
    cpu_addresses = []
    for pk_bytes in privkeys_bytes:
        k_int = int.from_bytes(pk_bytes, byteorder='little')
        for offset in range(keys_per_thread):
            cur_k = k_int + offset
            cur_bytes_be = cur_k.to_bytes(32, byteorder='big')
            cur_bytes_le = cur_bytes_be[::-1]
            addr = cpu_eth_address(cur_bytes_le)
            cpu_addresses.append(addr)

    # GPU
    privkeys_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=privkeys_np)
    addresses_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=total_addrs * 42)
    out_privkeys_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=total_addrs * 32)

    print(f"  Запуск kernel для {num_tests} базовых ключей x {keys_per_thread} = {total_addrs} адресов...")
    t_run = time.time()
    program.test_incremental(queue, (num_tests,), None,
                             privkeys_buf, addresses_buf, out_privkeys_buf,
                             np.int32(keys_per_thread), np.int32(num_tests))

    gpu_addresses_raw = np.empty(total_addrs * 42, dtype=np.uint8)
    cl.enqueue_copy(queue, gpu_addresses_raw, addresses_buf)
    queue.finish()
    print(f"  Kernel выполнен: {time.time() - t_run:.1f}с")

    # Compare
    matches = 0
    mismatches_shown = 0
    for i in range(total_addrs):
        gpu_addr = bytes(gpu_addresses_raw[i * 42:(i + 1) * 42]).decode('ascii', errors='replace')
        cpu_addr = cpu_addresses[i]
        if gpu_addr == cpu_addr:
            matches += 1
        elif mismatches_shown < 5:
            base_idx = i // keys_per_thread
            offset = i % keys_per_thread
            print(f"  MISMATCH [base={base_idx}, offset={offset}]:")
            print(f"    CPU: {cpu_addr}")
            print(f"    GPU: {gpu_addr}")
            mismatches_shown += 1

    if matches == total_addrs:
        print(f"  OK: {total_addrs}/{total_addrs} incremental addresses match")
        return True
    else:
        print(f"  FAIL: {matches}/{total_addrs} match ({total_addrs - matches} mismatches)")
        return False


# ============================================================
# Test 7: Batch inversion kernel
# ============================================================
def test_batch(ctx, queue, device, num_tests=100):
    """Test batch kernel: k, k+1, k+2, ... produce correct addresses (batch=4)."""
    print(f"\n{'='*60}")
    print(f"  Test 7: Batch inversion kernel (GPU vs CPU)")
    print(f"{'='*60}")

    # Number of sequential keys per base key (must be multiple of 4)
    keys_per_thread = 16

    source = build_source("prng.cl", "murmurhash3.cl", "bloom.cl",
                          "secp256k1.cl", "keccak256.cl")
    source += """
__kernel void test_batch(
    __global const ulong *privkeys,     // num_tests * 4 ulongs (base keys)
    __global uchar *addresses,          // num_tests * keys_per_thread * 42 bytes
    __global uchar *out_privkeys,       // num_tests * keys_per_thread * 32 bytes
    const int keys_per_thread,
    const int count
) {
    int gid = get_global_id(0);
    if (gid >= count) return;

    // Load base private key
    u256 k;
    k[0] = (uint)(privkeys[gid * 4 + 0] & 0xFFFFFFFF);
    k[1] = (uint)(privkeys[gid * 4 + 0] >> 32);
    k[2] = (uint)(privkeys[gid * 4 + 1] & 0xFFFFFFFF);
    k[3] = (uint)(privkeys[gid * 4 + 1] >> 32);
    k[4] = (uint)(privkeys[gid * 4 + 2] & 0xFFFFFFFF);
    k[5] = (uint)(privkeys[gid * 4 + 2] >> 32);
    k[6] = (uint)(privkeys[gid * 4 + 3] & 0xFFFFFFFF);
    k[7] = (uint)(privkeys[gid * 4 + 3] >> 32);

    // Full scalar mult for base key
    u256 px, py, pz;
    scalar_mult_G(px, py, pz, k);

    int num_batches = keys_per_thread / 4;
    int out_idx = 0;

    for (int batch = 0; batch < num_batches; batch++) {
        u256 px0, py0, pz0, k0;
        u256 px1, py1, pz1, k1;
        u256 px2, py2, pz2, k2;
        u256 px3, py3, pz3, k3;

        // Point 0
        u256_copy(px0, px); u256_copy(py0, py); u256_copy(pz0, pz);
        u256_copy(k0, k);

        // Point 1: P + G
        u256 tx, ty, tz;
        u256_copy(tx, px); u256_copy(ty, py); u256_copy(tz, pz);
        point_add_mixed(px, py, pz, tx, ty, tz, SECP256K1_GX, SECP256K1_GY);
        u256_inc(k);
        u256_copy(px1, px); u256_copy(py1, py); u256_copy(pz1, pz);
        u256_copy(k1, k);

        // Point 2: P + 2G
        u256_copy(tx, px); u256_copy(ty, py); u256_copy(tz, pz);
        point_add_mixed(px, py, pz, tx, ty, tz, SECP256K1_GX, SECP256K1_GY);
        u256_inc(k);
        u256_copy(px2, px); u256_copy(py2, py); u256_copy(pz2, pz);
        u256_copy(k2, k);

        // Point 3: P + 3G
        u256_copy(tx, px); u256_copy(ty, py); u256_copy(tz, pz);
        point_add_mixed(px, py, pz, tx, ty, tz, SECP256K1_GX, SECP256K1_GY);
        u256_inc(k);
        u256_copy(px3, px); u256_copy(py3, py); u256_copy(pz3, pz);
        u256_copy(k3, k);

        // Batch inversion
        batch_mod_inv_4(pz0, pz1, pz2, pz3);

        // Process each point
        u256 keys_arr[4];
        u256_copy(keys_arr[0], k0);
        u256_copy(keys_arr[1], k1);
        u256_copy(keys_arr[2], k2);
        u256_copy(keys_arr[3], k3);

        u256 pxs[4], pys[4], pzs[4];
        u256_copy(pxs[0], px0); u256_copy(pys[0], py0); u256_copy(pzs[0], pz0);
        u256_copy(pxs[1], px1); u256_copy(pys[1], py1); u256_copy(pzs[1], pz1);
        u256_copy(pxs[2], px2); u256_copy(pys[2], py2); u256_copy(pzs[2], pz2);
        u256_copy(pxs[3], px3); u256_copy(pys[3], py3); u256_copy(pzs[3], pz3);

        for (int i = 0; i < 4; i++) {
            int slot = gid * keys_per_thread + out_idx;

            // Write privkey
            ulong cur_pk[4];
            u256_to_privkey(keys_arr[i], cur_pk);
            __global uchar *pk_out = &out_privkeys[slot * 32];
            for (int ii = 0; ii < 4; ii++)
                for (int j = 0; j < 8; j++)
                    pk_out[ii * 8 + j] = (uchar)(cur_pk[ii] >> (j * 8));

            // Write address
            uchar pubkey[64];
            jacobian_to_pubkey_zi(pxs[i], pys[i], pzs[i], pubkey);
            uchar hash[32];
            keccak256_64(pubkey, hash);
            uchar addr[42];
            hash_to_eth_address(hash, addr);
            __global uchar *addr_out = &addresses[slot * 42];
            for (int ii = 0; ii < 42; ii++)
                addr_out[ii] = addr[ii];

            out_idx++;
        }

        // Advance to next batch
        u256_copy(tx, px); u256_copy(ty, py); u256_copy(tz, pz);
        point_add_mixed(px, py, pz, tx, ty, tz, SECP256K1_GX, SECP256K1_GY);
        u256_inc(k);
    }
}
"""

    print("  Компиляция batch kernel...")
    t_compile = time.time()
    program = cl.Program(ctx, source).build()
    print(f"  Компиляция: {time.time() - t_compile:.1f}с")

    # Generate random base private keys
    privkeys_bytes = []
    privkeys_np = np.empty(num_tests * 4, dtype=np.uint64)
    for i in range(num_tests):
        pk_bytes = os.urandom(32)
        while pk_bytes == b'\x00' * 32:
            pk_bytes = os.urandom(32)
        privkeys_bytes.append(pk_bytes)
        vals = np.frombuffer(pk_bytes, dtype=np.uint64)
        privkeys_np[i * 4:(i + 1) * 4] = vals

    # CPU reference: compute address for k, k+1, ..., k+keys_per_thread-1
    total_addrs = num_tests * keys_per_thread
    cpu_addresses = []
    for pk_bytes in privkeys_bytes:
        k_int = int.from_bytes(pk_bytes, byteorder='little')
        for offset in range(keys_per_thread):
            cur_k = k_int + offset
            cur_bytes_be = cur_k.to_bytes(32, byteorder='big')
            cur_bytes_le = cur_bytes_be[::-1]
            addr = cpu_eth_address(cur_bytes_le)
            cpu_addresses.append(addr)

    # GPU
    privkeys_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=privkeys_np)
    addresses_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=total_addrs * 42)
    out_privkeys_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=total_addrs * 32)

    print(f"  Запуск kernel для {num_tests} базовых ключей x {keys_per_thread} = {total_addrs} адресов...")
    t_run = time.time()
    program.test_batch(queue, (num_tests,), None,
                       privkeys_buf, addresses_buf, out_privkeys_buf,
                       np.int32(keys_per_thread), np.int32(num_tests))

    gpu_addresses_raw = np.empty(total_addrs * 42, dtype=np.uint8)
    cl.enqueue_copy(queue, gpu_addresses_raw, addresses_buf)
    queue.finish()
    print(f"  Kernel выполнен: {time.time() - t_run:.1f}с")

    # Compare
    matches = 0
    mismatches_shown = 0
    for i in range(total_addrs):
        gpu_addr = bytes(gpu_addresses_raw[i * 42:(i + 1) * 42]).decode('ascii', errors='replace')
        cpu_addr = cpu_addresses[i]
        if gpu_addr == cpu_addr:
            matches += 1
        elif mismatches_shown < 5:
            base_idx = i // keys_per_thread
            offset = i % keys_per_thread
            print(f"  MISMATCH [base={base_idx}, offset={offset}]:")
            print(f"    CPU: {cpu_addr}")
            print(f"    GPU: {gpu_addr}")
            mismatches_shown += 1

    if matches == total_addrs:
        print(f"  OK: {total_addrs}/{total_addrs} batch addresses match")
        return True
    else:
        print(f"  FAIL: {matches}/{total_addrs} match ({total_addrs - matches} mismatches)")
        return False


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="GPU Correctness Tests")
    parser.add_argument("--num-keys", type=int, default=100,
                        help="Number of keys to test (default: 100)")
    parser.add_argument("--test", type=str, default=None,
                        choices=["mmh3", "keccak", "secp256k1", "pipeline", "bloom", "incremental", "batch"],
                        help="Run specific test only")
    args = parser.parse_args()

    platform, device = select_gpu()
    print(f"OpenCL устройство: {device.name} ({platform.name})")

    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)

    results = {}

    tests = {
        "mmh3": lambda: test_mmh3(ctx, queue, device, args.num_keys),
        "keccak": lambda: test_keccak(ctx, queue, device, args.num_keys),
        "secp256k1": lambda: test_secp256k1(ctx, queue, device, args.num_keys),
        "pipeline": lambda: test_full_pipeline(ctx, queue, device, args.num_keys),
        "bloom": lambda: test_bloom(ctx, queue, device, args.num_keys),
        "incremental": lambda: test_incremental(ctx, queue, device, args.num_keys),
        "batch": lambda: test_batch(ctx, queue, device, args.num_keys),
    }

    if args.test:
        tests_to_run = {args.test: tests[args.test]}
    else:
        tests_to_run = tests

    for name, test_fn in tests_to_run.items():
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Summary
    print(f"\n{'='*60}")
    print(f"  ИТОГО")
    print(f"{'='*60}")
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:15s}: {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print(f"\n  Все тесты пройдены!")
    else:
        print(f"\n  Есть ошибки! Проверьте реализацию.")
    print(f"{'='*60}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
