"""
GPU Ethereum Wallet Generator + Checker (OpenCL)

Использует RTX 3080 Ti для генерации и проверки Ethereum адресов.
Bloom filter загружается в VRAM, весь pipeline выполняется на GPU.
На CPU возвращаются только bloom hits (~1%) для SQLite подтверждения.

Использование:
  python gpu_generator.py                          # дефолтные параметры
  python gpu_generator.py --global-size 262144     # кол-во GPU потоков
  python gpu_generator.py --keys-per-thread 16     # ключей на поток
  python gpu_generator.py --local-size 256         # размер workgroup
"""

import argparse
import json
import os
import signal
import sqlite3
import sys
import time
from datetime import datetime

import numpy as np

try:
    import pyopencl as cl
except ImportError:
    print("ОШИБКА: pyopencl не установлен. Установите:")
    print("  pip install pyopencl")
    sys.exit(1)

# --- Пути ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KERNELS_DIR = os.path.join(BASE_DIR, "kernels")
DB_PATH = os.path.join(BASE_DIR, "data", "eth_addresses.db")
BLOOM_PATH = os.path.join(BASE_DIR, "data", "bloom.bin")
META_PATH = os.path.join(BASE_DIR, "data", "bloom_meta.json")
FOUND_FILE = os.path.join(BASE_DIR, "found.txt")

# --- Конфигурация ---
MAX_HITS = 65536  # Максимум bloom hits за одну итерацию


def load_bloom_meta():
    with open(META_PATH, "r") as f:
        return json.load(f)


def save_found(address, private_key_hex, eth_balance, attempt_num):
    """Сохраняет найденный адрес в файл и SQLite."""
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
        print(f"[!] Ошибка записи в SQLite: {e}")


def build_kernel_source():
    """Собирает исходный код kernel из отдельных .cl файлов."""
    # Порядок важен: зависимости первыми
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
    """Выбирает GPU устройство для OpenCL."""
    platforms = cl.get_platforms()
    for platform in platforms:
        devices = platform.get_devices(device_type=cl.device_type.GPU)
        if devices:
            # Предпочитаем NVIDIA
            for dev in devices:
                if "nvidia" in dev.vendor.lower() or "nvidia" in platform.name.lower():
                    return platform, dev
            return platform, devices[0]

    print("ОШИБКА: GPU устройство не найдено.")
    print("\nДоступные платформы:")
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
        return f"{n_bytes / 1024 / 1024 / 1024:.1f} ГБ"
    if n_bytes >= 1024 * 1024:
        return f"{n_bytes / 1024 / 1024:.1f} МБ"
    if n_bytes >= 1024:
        return f"{n_bytes / 1024:.1f} КБ"
    return f"{n_bytes} Б"


def main():
    parser = argparse.ArgumentParser(description="GPU Ethereum Wallet Generator + Checker")
    parser.add_argument("--global-size", type=int, default=262144,
                        help="Кол-во GPU потоков (default: 262144 = 256K)")
    parser.add_argument("--local-size", type=int, default=256,
                        help="Размер workgroup (default: 256)")
    parser.add_argument("--keys-per-thread", type=int, default=256,
                        help="Ключей на поток за итерацию (default: 256)")
    parser.add_argument("--no-incremental", action="store_true",
                        help="Использовать старый kernel (полный scalar mult для каждого ключа)")
    parser.add_argument("--no-batch", action="store_true",
                        help="Использовать incremental без batch inversion (1 mod_inv на ключ)")
    args = parser.parse_args()

    # Проверка файлов
    for path, name in [(DB_PATH, "SQLite база"), (BLOOM_PATH, "Bloom filter"),
                       (META_PATH, "Bloom метаданные")]:
        if not os.path.exists(path):
            print(f"ОШИБКА: {name} не найден: {path}")
            print("Сначала запустите bloom_build.py для построения Bloom filter.")
            sys.exit(1)

    # Загрузка метаданных Bloom
    meta = load_bloom_meta()
    bloom_m = meta["m"]
    bloom_k = meta["k"]

    # Выбор GPU
    platform, device = select_gpu_device()
    vram_bytes = device.global_mem_size

    print("=" * 70)
    print("  GPU Ethereum Wallet Generator + Checker")
    print("=" * 70)
    print(f"  GPU:               {device.name}")
    print(f"  Platform:          {platform.name}")
    print(f"  VRAM:              {format_size(vram_bytes)}")
    print(f"  Max work group:    {device.max_work_group_size}")
    print(f"  Compute units:     {device.max_compute_units}")
    print(f"  Адресов в базе:    {meta['n']:,}")
    print(f"  Bloom filter:      {os.path.getsize(BLOOM_PATH) / 1024 / 1024:.1f} МБ "
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

    # Расчёт VRAM
    bloom_size = os.path.getsize(BLOOM_PATH)
    seeds_size = args.global_size * 4 * 8  # 4 ulongs per thread
    hits_pk_size = MAX_HITS * 32
    hits_addr_size = MAX_HITS * 42
    hit_count_size = 4
    total_vram = bloom_size + seeds_size + hits_pk_size + hits_addr_size + hit_count_size

    print(f"\n  VRAM бюджет:")
    print(f"    Bloom filter:    {format_size(bloom_size)}")
    print(f"    Seeds:           {format_size(seeds_size)}")
    print(f"    Hit buffers:     {format_size(hits_pk_size + hits_addr_size)}")
    print(f"    Итого:           {format_size(total_vram)} / {format_size(vram_bytes)}")

    if total_vram > vram_bytes * 0.9:
        print(f"\nВНИМАНИЕ: Не хватает VRAM! Уменьшите --global-size")
        sys.exit(1)

    print("=" * 70)
    print("  Нажмите Ctrl+C для остановки")
    print("=" * 70)
    print()

    # Создание OpenCL контекста
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    # Компиляция kernel
    print("Компиляция OpenCL kernel...")
    kernel_source = build_kernel_source()

    # Убираем #include из eth_generator.cl (мы уже конкатенировали)
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
        print(f"ОШИБКА компиляции kernel:")
        print(e)
        if hasattr(program, 'get_build_info'):
            log = program.get_build_info(device, cl.program_build_info.LOG)
            print(f"Build log:\n{log}")
        sys.exit(1)

    print("Kernel скомпилирован успешно!")

    # Загрузка Bloom filter в VRAM
    print("Загрузка Bloom filter в VRAM...")
    with open(BLOOM_PATH, "rb") as f:
        bloom_data = f.read()
    bloom_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                          hostbuf=bloom_data)
    del bloom_data  # Освобождаем RAM
    print(f"Bloom filter загружен в VRAM ({format_size(bloom_size)})")

    # Создание буферов
    seeds_np = np.empty(args.global_size * 4, dtype=np.uint64)
    seeds_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY, size=seeds_np.nbytes)

    hit_count_np = np.zeros(1, dtype=np.uint32)
    hit_count_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=4)

    hit_privkeys_np = np.empty(MAX_HITS * 32, dtype=np.uint8)
    hit_privkeys_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=hit_privkeys_np.nbytes)

    hit_addresses_np = np.empty(MAX_HITS * 42, dtype=np.uint8)
    hit_addresses_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=hit_addresses_np.nbytes)

    # SQLite соединение для проверки bloom hits
    db_conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    db_conn.execute("PRAGMA cache_size = -200000")  # ~200 МБ кеш
    db_cur = db_conn.cursor()

    # Получаем kernel
    if args.no_incremental:
        kernel = program.generate_and_check
    elif args.no_batch:
        kernel = program.generate_and_check_incremental
    else:
        kernel = program.generate_and_check_batch

    # Статистика
    total_keys = 0
    total_found = 0
    total_bloom_hits = 0
    t0 = time.time()
    prev_keys = 0
    prev_time = t0

    stop = False

    def signal_handler(sig, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, signal_handler)

    print(f"Запуск генерации на {device.name}...\n")

    iteration = 0
    while not stop:
        iteration += 1

        # 1. Генерация случайных seeds на CPU
        seeds_bytes = os.urandom(args.global_size * 32)
        seeds_np = np.frombuffer(seeds_bytes, dtype=np.uint64)

        # 2. Загрузка seeds в VRAM
        cl.enqueue_copy(queue, seeds_buf, seeds_np)

        # 3. Сброс счётчика hits
        hit_count_np[0] = 0
        cl.enqueue_copy(queue, hit_count_buf, hit_count_np)

        # 4. Запуск kernel
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

        # 5. Чтение результатов
        cl.enqueue_copy(queue, hit_count_np, hit_count_buf)
        queue.finish()

        n_hits = int(hit_count_np[0])
        keys_this_iter = args.global_size * args.keys_per_thread
        total_keys += keys_this_iter
        total_bloom_hits += n_hits

        # 6. Обработка bloom hits на CPU
        if n_hits > 0:
            actual_hits = min(n_hits, MAX_HITS)
            # Читаем только нужное количество
            pk_read = np.empty(MAX_HITS * 32, dtype=np.uint8)
            addr_read = np.empty(MAX_HITS * 42, dtype=np.uint8)
            cl.enqueue_copy(queue, pk_read, hit_privkeys_buf)
            cl.enqueue_copy(queue, addr_read, hit_addresses_buf)
            queue.finish()

            for i in range(actual_hits):
                privkey_bytes_le = bytes(pk_read[i * 32:(i + 1) * 32])
                privkey_bytes = privkey_bytes_le[::-1]  # Convert LE -> BE for standard format
                addr_str = bytes(addr_read[i * 42:(i + 1) * 42]).decode('ascii', errors='replace')

                # SQLite подтверждение
                db_cur.execute("SELECT eth_balance FROM addresses WHERE address = ?",
                               (addr_str,))
                row = db_cur.fetchone()
                if row is not None:
                    eth_balance = row[0]
                    privkey_hex = "0x" + privkey_bytes.hex()

                    print(f"\n{'=' * 60}")
                    print(f"[!!!] НАЙДЕН АДРЕС: {addr_str}")
                    print(f"[!!!] Приватный ключ: {privkey_hex}")
                    print(f"[!!!] Баланс ETH: {eth_balance}")
                    print(f"[!!!] Попытка #: {total_keys}")
                    print(f"{'=' * 60}\n")

                    save_found(addr_str, privkey_hex, eth_balance, total_keys)
                    total_found += 1

        # 7. Статистика (каждые ~5 секунд или каждую итерацию)
        now = time.time()
        if now - prev_time >= 5.0 or iteration == 1:
            elapsed = now - t0
            delta_keys = total_keys - prev_keys
            delta_time = now - prev_time

            rate = delta_keys / delta_time if delta_time > 0 else 0
            avg_rate = total_keys / elapsed if elapsed > 0 else 0
            bloom_hit_pct = (total_bloom_hits / total_keys * 100) if total_keys > 0 else 0

            print(f"[{elapsed:7.0f}с] Проверено: {format_number(total_keys):>10s} | "
                  f"Скорость: {format_number(rate):>8s} keys/sec | "
                  f"Avg: {format_number(avg_rate):>8s}/sec | "
                  f"Bloom hits: {total_bloom_hits} ({bloom_hit_pct:.3f}%) | "
                  f"Найдено: {total_found}")

            prev_keys = total_keys
            prev_time = now

    # Завершение
    db_conn.close()

    elapsed = time.time() - t0
    print()
    print("=" * 70)
    print("  ИТОГО (GPU)")
    print("=" * 70)
    print(f"  GPU:            {device.name}")
    print(f"  Время:          {elapsed:.1f}с")
    print(f"  Проверено:      {total_keys:,} ключей")
    if elapsed > 0:
        print(f"  Avg скорость:   {total_keys / elapsed:,.0f} keys/sec")
    print(f"  Bloom hits:     {total_bloom_hits:,} ({total_bloom_hits / total_keys * 100:.3f}%)"
          if total_keys > 0 else "")
    print(f"  Найдено:        {total_found}")
    if total_found > 0:
        print(f"  Результаты:     {FOUND_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
