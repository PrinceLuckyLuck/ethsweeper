
# ETHSWEEPER

`ethsweeper` is ethereum wallet generator and checker. Designed to search for used wallets on the Ethereum network at maximum speed.

You can try your luck at finding a wallet, at least one that doesn't have a zero balance, but the odds of winning the lottery are MUCH higher.

The odds of winning Powerball are 0.0000003%.

The chance of getting a wallet from the list is 0.0000000000000000000000000000000000000000281%.

This project was created to explore the possibilities of Claude Code. I have no idea what is written in OpenCL C.
 
---

# About

High-performance Ethereum private key generator with address lookup against ~411M known addresses from Google BigQuery. Two modes: CPU (multiprocessing) and GPU (OpenCL).

## How It Works

```
BigQuery (411M addresses) → SQLite (eth_addresses.db, ~41 GB)
                                ↓
                           Bloom filter (bloom.bin, ~470 MB)
                                ↓
                Generator (CPU or GPU)
                os.urandom → secp256k1 → keccak256 → address
                                ↓
                Bloom filter → "no" (99%) → next key
                             → "maybe"  → SQLite confirm → save to found.txt
```

Random private keys are generated, converted to Ethereum addresses via elliptic curve multiplication (secp256k1) and Keccak-256 hashing, then checked against a Bloom filter. The ~1% false positives are verified against SQLite. Confirmed matches are saved with their private key and balance.

## Performance

| Mode | Speed | Description |
|---|---|---|
| **GPU Batch** (default) | ~25-35M keys/sec | Montgomery batch inversion, 4 keys per mod_inv |
| **GPU Incremental** | ~8-10M keys/sec | Single scalar mult + point additions |
| **GPU Legacy** | ~3-5M keys/sec | Full scalar multiplication per key |
| **CPU** | ~30-80K keys/sec | Python multiprocessing |

Benchmarked on NVIDIA GeForce RTX 3080 Ti (12 GB VRAM, 80 SM).

## Requirements

- Python 3.12+
- NVIDIA GPU with OpenCL support (for GPU mode)
- Google Cloud service account with BigQuery access (for initial data load)
- ~42 GB disk space (SQLite database + Bloom filter)

## Setup

### 1. Clone and create virtual environment

```bash
git clone https://github.com/PrinceLuckyLuck/ethsweeper.git
cd ethsweeper
python -m venv venv
source venv/Scripts/activate  # Git Bash / MINGW
# or: venv\Scripts\activate   # Windows CMD
# or: source venv/bin/activate # Linux / macOS
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```
**Or** install manually:
```bash
pip install coincurve pycryptodome mmh3 numpy pyopencl
pip install google-cloud-bigquery-storage pyarrow  # only for init_db.py
```

### 3. Load addresses from BigQuery

Place your Google Cloud service account credentials at `data/credentials.json`, then:

```bash
python init_db.py
```

This downloads ~411M Ethereum addresses with balances into `data/eth_addresses.db` (~41 GB). One-time operation.

### 4. Build Bloom filter

```bash
python bloom_build.py
```

Generates `data/bloom.bin` (~470 MB) and `data/bloom_meta.json`. One-time operation.

### 5. Run

**GPU mode (recommended):**

```bash
# Batch inversion (default, fastest)
python gpu_generator.py

# Custom settings
python gpu_generator.py --global-size 262144 --keys-per-thread 256 --local-size 256

# Incremental mode (no batch inversion)
python gpu_generator.py --no-batch

# Legacy mode (full scalar mult per key)
python gpu_generator.py --no-incremental
```

**CPU mode:**

```bash
python generator.py                # workers = cpu_count / 2
python generator.py --workers 16   # custom worker count
```

**Stop:** `Ctrl+C` (graceful shutdown).

### 6. Verify GPU correctness

```bash
python test_gpu_correctness.py              # all 7 tests
python test_gpu_correctness.py --test batch # specific test
```

## GPU Parameters

| Parameter | Default | Description |
|---|---|---|
| `--global-size` | 262144 | Number of GPU threads |
| `--local-size` | 256 | Workgroup size |
| `--keys-per-thread` | 256 | Keys generated per thread per iteration |
| `--no-batch` | off | Use incremental mode without batch inversion |
| `--no-incremental` | off | Use legacy mode (full scalar mult) |

VRAM usage: Bloom (470 MB) + Seeds + Hit buffers = ~480-510 MB.

## Project Structure

```
├── generator.py              # CPU generator (multiprocessing)
├── gpu_generator.py          # GPU generator (PyOpenCL)
├── init_db.py                # BigQuery → SQLite loader
├── bloom_build.py            # Bloom filter builder
├── test_gpu_correctness.py   # GPU vs CPU correctness tests
├── kernels/
│   ├── eth_generator.cl      # Main OpenCL kernel
│   ├── secp256k1.cl          # Elliptic curve arithmetic (256-bit)
│   ├── keccak256.cl          # Keccak-256 hash (Ethereum variant)
│   ├── murmurhash3.cl        # MurmurHash3 for Bloom filter
│   ├── bloom.cl              # Bloom filter check on GPU
│   └── prng.cl               # xoshiro256** PRNG
└── data/
    ├── eth_addresses.db      # SQLite database (~41 GB, not in repo)
    ├── bloom.bin             # Bloom filter (~470 MB, not in repo)
    ├── bloom_meta.json       # Bloom filter metadata
    └── credentials.json      # GCP service account (not in repo)
```

## Technical Details

- **Bloom filter:** 3.94 billion bits, 7 hash functions (MurmurHash3), FPR ~1%
- **secp256k1:** 256-bit modular arithmetic with Jacobian coordinates, fast reduction (p = 2^256 - 2^32 - 977)
- **Keccak-256:** Ethereum variant (padding 0x01, not SHA-3's 0x06)
- **Batch inversion:** Montgomery's trick — 4 modular inversions reduced to 1, cutting cost from ~310 to ~77 mod_mul per key
- **PRNG:** xoshiro256** seeded from `os.urandom` on CPU, each GPU thread gets independent state
- **Private keys:** Little-endian on GPU (4 x ulong), converted to big-endian for output

## Output

Found matches are saved to:
- `found.txt` — text log with private key, address, and balance
- `data/eth_addresses.db` → `found` table — structured storage with timestamp

## License

MIT
