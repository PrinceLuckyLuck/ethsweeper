"""
Построение Bloom filter из SQLite базы адресов.
Читает ~411M адресов, строит bit array и сохраняет в data/bloom.bin + data/bloom_meta.json.

Параметры:
  FPR ≈ 1% → m = n * 9.6 bits, k = 7 хеш-функций (mmh3 с разными seed)
"""

import json
import math
import mmap
import os
import sqlite3
import struct
import sys
import time

import mmh3

# --- Конфигурация ---
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "eth_addresses.db")
BLOOM_PATH = os.path.join(os.path.dirname(__file__), "data", "bloom.bin")
META_PATH = os.path.join(os.path.dirname(__file__), "data", "bloom_meta.json")
NUM_HASHES = 7  # k = 7
FPR_TARGET = 0.01  # 1%
BATCH_SIZE = 500_000


def compute_bloom_size(n, fpr):
    """Вычисляет оптимальный размер Bloom filter в битах."""
    m = -n * math.log(fpr) / (math.log(2) ** 2)
    return int(math.ceil(m))


def bloom_add(buf, address_bytes, m, k):
    """Добавляет элемент в Bloom filter (bytearray buf)."""
    for seed in range(k):
        h = mmh3.hash(address_bytes, seed, signed=False) % m
        byte_idx = h >> 3
        bit_idx = h & 7
        buf[byte_idx] |= (1 << bit_idx)


def bloom_check(buf, address_bytes, m, k):
    """Проверяет наличие элемента в Bloom filter."""
    for seed in range(k):
        h = mmh3.hash(address_bytes, seed, signed=False) % m
        byte_idx = h >> 3
        bit_idx = h & 7
        if not (buf[byte_idx] & (1 << bit_idx)):
            return False
    return True


def main():
    if not os.path.exists(DB_PATH):
        print(f"ОШИБКА: База данных не найдена: {DB_PATH}")
        sys.exit(1)

    # Подсчёт адресов
    print("Подключение к SQLite...")
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.execute("PRAGMA cache_size = -2000000")  # ~2 ГБ кеш
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM addresses")
    n = cur.fetchone()[0]
    print(f"Адресов в базе: {n:,}")

    # Вычисление параметров
    m = compute_bloom_size(n, FPR_TARGET)
    m_bytes = (m + 7) // 8
    m = m_bytes * 8  # округляем до целого байта

    print(f"Параметры Bloom filter:")
    print(f"  m = {m:,} бит ({m_bytes / 1024 / 1024:.1f} МБ)")
    print(f"  k = {NUM_HASHES} хеш-функций")
    print(f"  Ожидаемый FPR = {(1 - math.exp(-NUM_HASHES * n / m)) ** NUM_HASHES:.6f}")

    # Создаём файл нужного размера
    print(f"\nСоздание файла {BLOOM_PATH} ({m_bytes / 1024 / 1024:.1f} МБ)...")
    os.makedirs(os.path.dirname(BLOOM_PATH), exist_ok=True)

    with open(BLOOM_PATH, "wb") as f:
        # Заполняем нулями блоками по 1 МБ
        block = b'\x00' * (1024 * 1024)
        full_blocks = m_bytes // len(block)
        remainder = m_bytes % len(block)
        for _ in range(full_blocks):
            f.write(block)
        if remainder:
            f.write(b'\x00' * remainder)

    # Открываем через mmap для записи
    print("Заполнение Bloom filter...")
    fd = os.open(BLOOM_PATH, os.O_RDWR)
    try:
        mm = mmap.mmap(fd, m_bytes, access=mmap.ACCESS_WRITE)
        buf = mm

        count = 0
        t0 = time.time()
        t_last = t0

        cur.execute("SELECT address FROM addresses")
        while True:
            rows = cur.fetchmany(BATCH_SIZE)
            if not rows:
                break
            for (addr,) in rows:
                addr_bytes = addr.encode('ascii') if isinstance(addr, str) else addr
                bloom_add(buf, addr_bytes, m, NUM_HASHES)
                count += 1

            now = time.time()
            if now - t_last >= 5.0:
                elapsed = now - t0
                rate = count / elapsed
                pct = count / n * 100
                print(f"  {count:>12,} / {n:,} ({pct:.1f}%) | {rate:,.0f} addr/sec | "
                      f"прошло {elapsed:.0f}с")
                t_last = now

        mm.flush()
        mm.close()
    finally:
        os.close(fd)

    elapsed = time.time() - t0
    print(f"\nГотово! Обработано {count:,} адресов за {elapsed:.1f}с ({count/elapsed:,.0f} addr/sec)")

    # Верификация: проверяем несколько известных адресов
    print("\nВерификация...")
    fd = os.open(BLOOM_PATH, os.O_RDONLY)
    try:
        mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)

        cur.execute("SELECT address FROM addresses LIMIT 5")
        test_addrs = [row[0] for row in cur.fetchall()]

        ok = 0
        for addr in test_addrs:
            found = bloom_check(mm, addr.encode('ascii'), m, NUM_HASHES)
            status = "OK" if found else "MISS!"
            print(f"  {addr}: {status}")
            if found:
                ok += 1

        # Проверяем заведомо несуществующий адрес
        fake = "0x0000000000000000000000000000000000000000"
        fake_found = bloom_check(mm, fake.encode('ascii'), m, NUM_HASHES)
        print(f"  {fake} (fake): {'FOUND (FP)' if fake_found else 'NOT FOUND (OK)'}")

        mm.close()
    finally:
        os.close(fd)

    print(f"\nВерификация: {ok}/{len(test_addrs)} известных адресов найдены в Bloom filter")
    if ok < len(test_addrs):
        print("ВНИМАНИЕ: не все адреса найдены! Возможна ошибка в построении.")

    # Сохранение метаданных
    meta = {
        "m": m,
        "m_bytes": m_bytes,
        "k": NUM_HASHES,
        "n": count,
        "fpr_target": FPR_TARGET,
        "fpr_actual": (1 - math.exp(-NUM_HASHES * count / m)) ** NUM_HASHES,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Метаданные сохранены: {META_PATH}")
    print(f"Bloom filter: {BLOOM_PATH} ({os.path.getsize(BLOOM_PATH) / 1024 / 1024:.1f} МБ)")


if __name__ == "__main__":
    main()
