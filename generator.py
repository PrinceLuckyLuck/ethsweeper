"""
Ethereum Wallet Generator + Checker

Генерирует случайные приватные ключи, вычисляет адреса,
проверяет через Bloom filter + SQLite среди ~411M известных адресов.

Использование:
  python generator.py              # запуск с дефолтными настройками (30 workers)
  python generator.py --workers 16 # указать кол-во workers
  python generator.py --workers 16 --batch 2000  # batch размер для статистики
"""

import argparse
import json
import mmap
import multiprocessing
import os
import signal
import sqlite3
import sys
import time
from datetime import datetime

import mmh3
from coincurve import PrivateKey
from Crypto.Hash import keccak



# --- Пути ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "eth_addresses.db")
BLOOM_PATH = os.path.join(BASE_DIR, "data", "bloom.bin")
META_PATH = os.path.join(BASE_DIR, "data", "bloom_meta.json")
FOUND_FILE = os.path.join(BASE_DIR, "found.txt")


def load_bloom_meta():
    with open(META_PATH, "r") as f:
        return json.load(f)


def bloom_check(mm, address_bytes, m, k):
    """Проверяет наличие элемента в Bloom filter через mmap."""
    for seed in range(k):
        h = mmh3.hash(address_bytes, seed, signed=False) % m
        byte_idx = h >> 3
        bit_idx = h & 7
        if not (mm[byte_idx] & (1 << bit_idx)):
            return False
    return True


def generate_eth_address(privkey_bytes):
    """Генерирует Ethereum адрес из 32 байт приватного ключа."""
    pk = PrivateKey(privkey_bytes)
    # uncompressed public key (65 bytes: 04 + x + y), берём без префикса 04
    pubkey = pk.public_key.format(compressed=False)[1:]
    # keccak256
    h = keccak.new(digest_bits=256)
    h.update(pubkey)
    addr_bytes = h.digest()[-20:]
    return "0x" + addr_bytes.hex()


def save_found(address, private_key_hex, eth_balance, attempt_num, db_path):
    """Сохраняет найденный адрес в файл и SQLite."""
    timestamp = datetime.now().isoformat()
    line = f"{timestamp} | {address} | {private_key_hex} | balance={eth_balance}\n"

    # Запись в файл
    with open(FOUND_FILE, "a") as f:
        f.write(line)

    # Запись в SQLite
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
        print(f"[!] Ошибка записи в SQLite: {e}")


def worker(worker_id, bloom_path, bloom_m, bloom_k, db_path,
           counter, found_counter, stop_event, batch_size):
    """Worker-процесс: генерирует ключи и проверяет адреса."""
    # Открываем Bloom filter через mmap (read-only, ОС шарит страницы)
    fd = os.open(bloom_path, os.O_RDONLY)
    mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)

    # SQLite-соединение (read-only)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA cache_size = -100000")  # ~100 МБ кеш на worker
    cur = conn.cursor()

    local_count = 0
    bloom_hits = 0

    try:
        while not stop_event.is_set():
            for _ in range(batch_size):
                # 1. Генерация случайного приватного ключа
                privkey_bytes = os.urandom(32)

                # 2-3. Вычисление Ethereum адреса
                address = generate_eth_address(privkey_bytes)

                # 4. Проверка в Bloom filter
                addr_encoded = address.encode('ascii')
                if bloom_check(mm, addr_encoded, bloom_m, bloom_k):
                    bloom_hits += 1
                    # 5. Подтверждение в SQLite
                    cur.execute("SELECT eth_balance FROM addresses WHERE address = ?",
                                (address,))
                    row = cur.fetchone()
                    if row is not None:
                        eth_balance = row[0]
                        privkey_hex = "0x" + privkey_bytes.hex()
                        total_so_far = counter.value + local_count

                        print(f"\n{'='*60}")
                        print(f"[!!!] НАЙДЕН АДРЕС: {address}")
                        print(f"[!!!] Приватный ключ: {privkey_hex}")
                        print(f"[!!!] Баланс ETH: {eth_balance}")
                        print(f"[!!!] Попытка #: {total_so_far}")
                        print(f"{'='*60}\n")

                        save_found(address, privkey_hex, eth_balance, total_so_far, db_path)

                        with found_counter.get_lock():
                            found_counter.value += 1

                local_count += 1

            # Периодически отправляем статистику
            with counter.get_lock():
                counter.value += batch_size
            local_count = 0

    except KeyboardInterrupt:
        pass
    finally:
        # Отправляем остаток
        if local_count > 0:
            with counter.get_lock():
                counter.value += local_count

        mm.close()
        os.close(fd)
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
                        help=f"Кол-во worker процессов (default: {max(1, os.cpu_count() // 2)})")
    parser.add_argument("--batch", type=int, default=1000,
                        help="Размер batch для обновления счётчика (default: 1000)")
    args = parser.parse_args()

    # Проверка файлов
    for path, name in [(DB_PATH, "SQLite база"), (BLOOM_PATH, "Bloom filter"),
                       (META_PATH, "Bloom метаданные")]:
        if not os.path.exists(path):
            print(f"ОШИБКА: {name} не найден: {path}")
            print("Сначала запустите bloom_build.py для построения Bloom filter.")
            sys.exit(1)

    # Загрузка метаданных
    meta = load_bloom_meta()
    bloom_m = meta["m"]
    bloom_k = meta["k"]

    print("=" * 60)
    print("  Ethereum Wallet Generator + Checker")
    print("=" * 60)
    print(f"  Адресов в базе:    {meta['n']:,}")
    print(f"  Bloom filter:      {os.path.getsize(BLOOM_PATH) / 1024 / 1024:.1f} МБ "
          f"(FPR={meta['fpr_actual']:.4%})")
    print(f"  Workers:           {args.workers}")
    print(f"  Batch size:        {args.batch}")
    print(f"  Найденные адреса:  {FOUND_FILE}")
    print("=" * 60)
    print("  Нажмите Ctrl+C для остановки")
    print("=" * 60)
    print()

    # Shared state
    counter = multiprocessing.Value('q', 0)       # unsigned long long
    found_counter = multiprocessing.Value('i', 0)  # int
    stop_event = multiprocessing.Event()

    # Запуск workers
    workers = []
    for i in range(args.workers):
        p = multiprocessing.Process(
            target=worker,
            args=(i, BLOOM_PATH, bloom_m, bloom_k, DB_PATH,
                  counter, found_counter, stop_event, args.batch),
            daemon=True
        )
        p.start()
        workers.append(p)

    print(f"Запущено {len(workers)} worker-процессов")
    print()

    # Мониторинг
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

            print(f"[{elapsed:7.0f}с] Проверено: {format_number(total):>10s} | "
                  f"Скорость: {rate:>10,.0f} keys/sec ({per_worker:,.0f}/worker) | "
                  f"Avg: {avg_rate:>10,.0f}/sec | "
                  f"Найдено: {found} | Workers: {alive}/{args.workers}")

            prev_count = total
            prev_time = now

            # Проверка что workers живы
            if alive == 0:
                print("\nВсе workers завершились!")
                break

    except KeyboardInterrupt:
        print("\n\nОстановка...")
        stop_event.set()

        # Ждём завершения workers (max 10 сек)
        deadline = time.time() + 10
        for p in workers:
            remaining = max(0.1, deadline - time.time())
            p.join(timeout=remaining)

        # Убиваем зависшие
        for p in workers:
            if p.is_alive():
                p.terminate()

    # Итоговая статистика
    total = counter.value
    found = found_counter.value
    elapsed = time.time() - t0

    print()
    print("=" * 60)
    print("  ИТОГО")
    print("=" * 60)
    print(f"  Время:          {elapsed:.1f}с")
    print(f"  Проверено:      {total:,} ключей")
    print(f"  Avg скорость:   {total / elapsed:,.0f} keys/sec" if elapsed > 0 else "")
    print(f"  Найдено:        {found}")
    if found > 0:
        print(f"  Результаты:     {FOUND_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
