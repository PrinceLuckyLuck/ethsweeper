"""
Загрузка всех Ethereum-адресов и балансов из BigQuery в SQLite.
Таблица: bigquery-public-data.crypto_ethereum.balances (~411M строк)
Используем BigQuery Storage Read API напрямую (gRPC, параллельные потоки).
"""

import sqlite3
import time
from google.cloud.bigquery_storage import BigQueryReadClient, types
from google.oauth2 import service_account

DB_PATH = "data/eth_addresses.db"

# --- BigQuery Storage Read API ---
credentials = service_account.Credentials.from_service_account_file("data/credentials.json")
storage_client = BigQueryReadClient(credentials=credentials)

table = "projects/bigquery-public-data/datasets/crypto_ethereum/tables/balances"

read_session = types.ReadSession(
    table=table,
    data_format=types.DataFormat.ARROW,
    read_options=types.ReadSession.TableReadOptions(
        selected_fields=["address", "eth_balance"],
    ),
)

parent = f"projects/{credentials.project_id}"
session = storage_client.create_read_session(
    parent=parent,
    read_session=read_session,
    max_stream_count=0,  # сервер выберет оптимальное количество потоков
)

print(f"Создано потоков чтения: {len(session.streams)}")

# --- SQLite ---
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute("PRAGMA page_size=32768")       # большие страницы — меньше I/O
cur.execute("PRAGMA journal_mode=WAL")
cur.execute("PRAGMA synchronous=OFF")
cur.execute("PRAGMA temp_store=MEMORY")     # временные данные в RAM (для индекса)
cur.execute("PRAGMA cache_size=-2000000")   # 2 ГБ кэш

cur.execute("DROP TABLE IF EXISTS addresses")
cur.execute("""
    CREATE TABLE addresses (
        address TEXT,
        eth_balance REAL
    )
""")

cur.execute("""
    CREATE TABLE IF NOT EXISTS found (
        address TEXT NOT NULL,
        private_key TEXT NOT NULL,
        eth_balance REAL,
        found_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        attempt_number INTEGER,
        UNIQUE(address)
    )
""")
conn.commit()

# --- Загрузка ---
print("Чтение через Storage Read API (gRPC)...")
start = time.time()
count = 0

for i, stream in enumerate(session.streams):
    reader = storage_client.read_rows(stream.name)

    for message in reader.rows().pages:
        tbl = message.to_arrow()
        addresses = tbl.column("address").to_pylist()
        balances = [float(b) if b is not None else 0.0 for b in tbl.column("eth_balance").to_pylist()]

        cur.executemany("INSERT INTO addresses VALUES (?, ?)", zip(addresses, balances))
        count += len(addresses)

        if count % 1_000_000 < len(addresses):
            conn.commit()
            elapsed = time.time() - start
            speed = count / elapsed if elapsed > 0 else 0
            print(f"  {count:>12,} адресов  |  поток {i+1}/{len(session.streams)}  |  {elapsed:.0f}с  |  {speed:,.0f} строк/с")

conn.commit()

# --- Индекс ---
elapsed = time.time() - start
print(f"\nЗагружено {count:,} адресов за {elapsed:.0f}с")
print("Создание индекса по address...")

cur.execute("CREATE UNIQUE INDEX idx_address ON addresses(address)")
conn.commit()

elapsed = time.time() - start
print(f"Индекс создан. Общее время: {elapsed:.0f}с")
print(f"БД сохранена: {DB_PATH}")

total = cur.execute("SELECT COUNT(*) FROM addresses").fetchone()[0]
print(f"Строк в таблице addresses: {total:,}")

conn.close()
