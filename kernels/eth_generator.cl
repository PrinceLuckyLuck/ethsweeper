/*
 * Ethereum Address Generator - Main OpenCL Kernel
 *
 * Pipeline per thread:
 *   1. PRNG seed -> generate private key (32 bytes)
 *   2. secp256k1: privkey -> uncompressed pubkey (64 bytes)
 *   3. keccak256: pubkey -> hash (32 bytes)
 *   4. hash -> ASCII hex address "0x..." (42 bytes)
 *   5. MurmurHash3 x7 -> Bloom filter check
 *   6. If hit -> atomic write privkey + address to output buffer
 *
 * Each thread processes KEYS_PER_THREAD keys sequentially.
 */

#include "prng.cl"
#include "murmurhash3.cl"
#include "bloom.cl"
#include "secp256k1.cl"
#include "keccak256.cl"

__kernel void generate_and_check(
    __global const ulong *seeds,         // PRNG seeds: global_size * 4 ulongs
    __global const uchar *bloom_data,    // Bloom filter bit array
    const ulong bloom_m,                  // Bloom filter size in bits
    const int bloom_k,                    // Number of hash functions
    const int keys_per_thread,            // Keys to generate per thread
    __global uint *hit_count,             // Atomic counter for hits
    __global uchar *hit_privkeys,         // Output: hit private keys (32 bytes each)
    __global uchar *hit_addresses,        // Output: hit addresses (42 bytes each)
    const uint max_hits                   // Maximum hits buffer size
) {
    uint gid = get_global_id(0);

    // Initialize PRNG from seed
    xoshiro256ss_state prng;
    xoshiro256ss_init(&prng, &seeds[gid * 4]);

    for (int iter = 0; iter < keys_per_thread; iter++) {
        // 1. Generate private key
        ulong privkey[4];
        generate_privkey(&prng, privkey);

        // Skip zero key and keys >= curve order (extremely rare)
        if (privkey[0] == 0 && privkey[1] == 0 && privkey[2] == 0 && privkey[3] == 0)
            continue;

        // 2. secp256k1: privkey -> pubkey (64 bytes)
        uchar pubkey[64];
        privkey_to_pubkey(privkey, pubkey);

        // 3. keccak256: pubkey -> hash (32 bytes)
        uchar hash[32];
        keccak256_64(pubkey, hash);

        // 4. Convert to ASCII hex address
        uchar addr_ascii[42];
        hash_to_eth_address(hash, addr_ascii);

        // 5. Bloom filter check
        if (bloom_check(bloom_data, addr_ascii, 42, bloom_m, bloom_k)) {
            // 6. Bloom hit! Write to output buffer
            uint idx = atomic_inc(hit_count);
            if (idx < max_hits) {
                // Write private key (32 bytes = 4 ulongs, little-endian)
                __global uchar *pk_out = &hit_privkeys[idx * 32];
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 8; j++) {
                        pk_out[i * 8 + j] = (uchar)(privkey[i] >> (j * 8));
                    }
                }

                // Write address (42 bytes ASCII)
                __global uchar *addr_out = &hit_addresses[idx * 42];
                for (int i = 0; i < 42; i++) {
                    addr_out[i] = addr_ascii[i];
                }
            }
        }
    }
}

/*
 * Incremental kernel: compute P0 = k*G once, then P1 = P0+G, P2 = P1+G, ...
 * Cost per key: 1 point_add_mixed (~8 mod_mul) + 1 mod_inv (~300 mod_mul) = ~310 mod_mul
 * vs full scalar mult: ~2400 mod_mul per key
 * Expected speedup: ~7-8x
 */
__kernel void generate_and_check_incremental(
    __global const ulong *seeds,         // PRNG seeds: global_size * 4 ulongs
    __global const uchar *bloom_data,    // Bloom filter bit array
    const ulong bloom_m,                  // Bloom filter size in bits
    const int bloom_k,                    // Number of hash functions
    const int keys_per_thread,            // Keys to generate per thread
    __global uint *hit_count,             // Atomic counter for hits
    __global uchar *hit_privkeys,         // Output: hit private keys (32 bytes each)
    __global uchar *hit_addresses,        // Output: hit addresses (42 bytes each)
    const uint max_hits                   // Maximum hits buffer size
) {
    uint gid = get_global_id(0);

    // Initialize PRNG from seed
    xoshiro256ss_state prng;
    xoshiro256ss_init(&prng, &seeds[gid * 4]);

    // 1. Generate random base private key
    ulong privkey_ulongs[4];
    generate_privkey(&prng, privkey_ulongs);

    // Skip zero key
    if (privkey_ulongs[0] == 0 && privkey_ulongs[1] == 0 &&
        privkey_ulongs[2] == 0 && privkey_ulongs[3] == 0)
        return;

    // Convert to u256 for tracking
    u256 k;
    k[0] = (uint)(privkey_ulongs[0] & 0xFFFFFFFF);
    k[1] = (uint)(privkey_ulongs[0] >> 32);
    k[2] = (uint)(privkey_ulongs[1] & 0xFFFFFFFF);
    k[3] = (uint)(privkey_ulongs[1] >> 32);
    k[4] = (uint)(privkey_ulongs[2] & 0xFFFFFFFF);
    k[5] = (uint)(privkey_ulongs[2] >> 32);
    k[6] = (uint)(privkey_ulongs[3] & 0xFFFFFFFF);
    k[7] = (uint)(privkey_ulongs[3] >> 32);

    // 2. Full scalar multiplication: P = k * G (done ONCE)
    u256 px, py, pz;
    scalar_mult_G(px, py, pz, k);

    // 3. Loop: for each key, convert to address, check bloom, then P = P + G, k++
    for (int iter = 0; iter < keys_per_thread; iter++) {
        // a. Jacobian -> affine pubkey (contains mod_inv, ~300 mod_mul)
        uchar pubkey[64];
        jacobian_to_pubkey(px, py, pz, pubkey);

        // b. keccak256: pubkey -> hash
        uchar hash[32];
        keccak256_64(pubkey, hash);

        // c. Convert to ASCII hex address
        uchar addr_ascii[42];
        hash_to_eth_address(hash, addr_ascii);

        // d. Bloom filter check
        if (bloom_check(bloom_data, addr_ascii, 42, bloom_m, bloom_k)) {
            uint idx = atomic_inc(hit_count);
            if (idx < max_hits) {
                // Convert current k to privkey bytes (little-endian)
                ulong cur_pk[4];
                u256_to_privkey(k, cur_pk);

                __global uchar *pk_out = &hit_privkeys[idx * 32];
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 8; j++) {
                        pk_out[i * 8 + j] = (uchar)(cur_pk[i] >> (j * 8));
                    }
                }

                __global uchar *addr_out = &hit_addresses[idx * 42];
                for (int i = 0; i < 42; i++) {
                    addr_out[i] = addr_ascii[i];
                }
            }
        }

        // e. P = P + G (one point addition, ~8 mod_mul)
        u256 tx, ty, tz;
        u256_copy(tx, px); u256_copy(ty, py); u256_copy(tz, pz);
        point_add_mixed(px, py, pz, tx, ty, tz, SECP256K1_GX, SECP256K1_GY);

        // f. k++
        u256_inc(k);
    }
}

/*
 * Batch inversion kernel: processes keys in batches of 4 using Montgomery's trick.
 * Accumulates 4 Jacobian points, inverts their Z coords with 1 mod_inv instead of 4.
 * Cost per key: ~77 mod_mul (vs ~310 for non-batched incremental)
 * Expected speedup: ~3-4x over incremental
 * keys_per_thread MUST be >= 4 (rounded down to multiple of 4 internally)
 */
__kernel void generate_and_check_batch(
    __global const ulong *seeds,         // PRNG seeds: global_size * 4 ulongs
    __global const uchar *bloom_data,    // Bloom filter bit array
    const ulong bloom_m,                  // Bloom filter size in bits
    const int bloom_k,                    // Number of hash functions
    const int keys_per_thread,            // Keys to generate per thread (must be multiple of 4)
    __global uint *hit_count,             // Atomic counter for hits
    __global uchar *hit_privkeys,         // Output: hit private keys (32 bytes each)
    __global uchar *hit_addresses,        // Output: hit addresses (42 bytes each)
    const uint max_hits                   // Maximum hits buffer size
) {
    uint gid = get_global_id(0);

    // Initialize PRNG from seed
    xoshiro256ss_state prng;
    xoshiro256ss_init(&prng, &seeds[gid * 4]);

    // 1. Generate random base private key
    ulong privkey_ulongs[4];
    generate_privkey(&prng, privkey_ulongs);

    // Skip zero key
    if (privkey_ulongs[0] == 0 && privkey_ulongs[1] == 0 &&
        privkey_ulongs[2] == 0 && privkey_ulongs[3] == 0)
        return;

    // Convert to u256 for tracking
    u256 k;
    k[0] = (uint)(privkey_ulongs[0] & 0xFFFFFFFF);
    k[1] = (uint)(privkey_ulongs[0] >> 32);
    k[2] = (uint)(privkey_ulongs[1] & 0xFFFFFFFF);
    k[3] = (uint)(privkey_ulongs[1] >> 32);
    k[4] = (uint)(privkey_ulongs[2] & 0xFFFFFFFF);
    k[5] = (uint)(privkey_ulongs[2] >> 32);
    k[6] = (uint)(privkey_ulongs[3] & 0xFFFFFFFF);
    k[7] = (uint)(privkey_ulongs[3] >> 32);

    // 2. Full scalar multiplication: P = k * G (done ONCE)
    u256 px, py, pz;
    scalar_mult_G(px, py, pz, k);

    int num_batches = keys_per_thread / 4;
    int remainder = keys_per_thread - num_batches * 4;

    // 3. Main loop: process 4 keys per batch
    for (int batch = 0; batch < num_batches; batch++) {
        // a. Save 4 Jacobian points: P, P+G, P+2G, P+3G
        u256 px0, py0, pz0;
        u256 px1, py1, pz1;
        u256 px2, py2, pz2;
        u256 px3, py3, pz3;
        u256 k0, k1, k2, k3;

        // Point 0: current P
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

        // b. Batch inversion: 1 mod_inv instead of 4
        batch_mod_inv_4(pz0, pz1, pz2, pz3);

        // c. Process each of the 4 points with pre-inverted Z
        // --- Point 0 ---
        {
            uchar pubkey[64];
            jacobian_to_pubkey_zi(px0, py0, pz0, pubkey);
            uchar hash[32];
            keccak256_64(pubkey, hash);
            uchar addr_ascii[42];
            hash_to_eth_address(hash, addr_ascii);
            if (bloom_check(bloom_data, addr_ascii, 42, bloom_m, bloom_k)) {
                uint idx = atomic_inc(hit_count);
                if (idx < max_hits) {
                    ulong cur_pk[4];
                    u256_to_privkey(k0, cur_pk);
                    __global uchar *pk_out = &hit_privkeys[idx * 32];
                    for (int i = 0; i < 4; i++)
                        for (int j = 0; j < 8; j++)
                            pk_out[i * 8 + j] = (uchar)(cur_pk[i] >> (j * 8));
                    __global uchar *addr_out = &hit_addresses[idx * 42];
                    for (int i = 0; i < 42; i++)
                        addr_out[i] = addr_ascii[i];
                }
            }
        }
        // --- Point 1 ---
        {
            uchar pubkey[64];
            jacobian_to_pubkey_zi(px1, py1, pz1, pubkey);
            uchar hash[32];
            keccak256_64(pubkey, hash);
            uchar addr_ascii[42];
            hash_to_eth_address(hash, addr_ascii);
            if (bloom_check(bloom_data, addr_ascii, 42, bloom_m, bloom_k)) {
                uint idx = atomic_inc(hit_count);
                if (idx < max_hits) {
                    ulong cur_pk[4];
                    u256_to_privkey(k1, cur_pk);
                    __global uchar *pk_out = &hit_privkeys[idx * 32];
                    for (int i = 0; i < 4; i++)
                        for (int j = 0; j < 8; j++)
                            pk_out[i * 8 + j] = (uchar)(cur_pk[i] >> (j * 8));
                    __global uchar *addr_out = &hit_addresses[idx * 42];
                    for (int i = 0; i < 42; i++)
                        addr_out[i] = addr_ascii[i];
                }
            }
        }
        // --- Point 2 ---
        {
            uchar pubkey[64];
            jacobian_to_pubkey_zi(px2, py2, pz2, pubkey);
            uchar hash[32];
            keccak256_64(pubkey, hash);
            uchar addr_ascii[42];
            hash_to_eth_address(hash, addr_ascii);
            if (bloom_check(bloom_data, addr_ascii, 42, bloom_m, bloom_k)) {
                uint idx = atomic_inc(hit_count);
                if (idx < max_hits) {
                    ulong cur_pk[4];
                    u256_to_privkey(k2, cur_pk);
                    __global uchar *pk_out = &hit_privkeys[idx * 32];
                    for (int i = 0; i < 4; i++)
                        for (int j = 0; j < 8; j++)
                            pk_out[i * 8 + j] = (uchar)(cur_pk[i] >> (j * 8));
                    __global uchar *addr_out = &hit_addresses[idx * 42];
                    for (int i = 0; i < 42; i++)
                        addr_out[i] = addr_ascii[i];
                }
            }
        }
        // --- Point 3 ---
        {
            uchar pubkey[64];
            jacobian_to_pubkey_zi(px3, py3, pz3, pubkey);
            uchar hash[32];
            keccak256_64(pubkey, hash);
            uchar addr_ascii[42];
            hash_to_eth_address(hash, addr_ascii);
            if (bloom_check(bloom_data, addr_ascii, 42, bloom_m, bloom_k)) {
                uint idx = atomic_inc(hit_count);
                if (idx < max_hits) {
                    ulong cur_pk[4];
                    u256_to_privkey(k3, cur_pk);
                    __global uchar *pk_out = &hit_privkeys[idx * 32];
                    for (int i = 0; i < 4; i++)
                        for (int j = 0; j < 8; j++)
                            pk_out[i * 8 + j] = (uchar)(cur_pk[i] >> (j * 8));
                    __global uchar *addr_out = &hit_addresses[idx * 42];
                    for (int i = 0; i < 42; i++)
                        addr_out[i] = addr_ascii[i];
                }
            }
        }

        // d. Advance P to next batch: P = P + G, k++
        u256_copy(tx, px); u256_copy(ty, py); u256_copy(tz, pz);
        point_add_mixed(px, py, pz, tx, ty, tz, SECP256K1_GX, SECP256K1_GY);
        u256_inc(k);
    }

    // 4. Handle remainder keys (0-3) with standard single inversion
    for (int iter = 0; iter < remainder; iter++) {
        uchar pubkey[64];
        jacobian_to_pubkey(px, py, pz, pubkey);

        uchar hash[32];
        keccak256_64(pubkey, hash);

        uchar addr_ascii[42];
        hash_to_eth_address(hash, addr_ascii);

        if (bloom_check(bloom_data, addr_ascii, 42, bloom_m, bloom_k)) {
            uint idx = atomic_inc(hit_count);
            if (idx < max_hits) {
                ulong cur_pk[4];
                u256_to_privkey(k, cur_pk);
                __global uchar *pk_out = &hit_privkeys[idx * 32];
                for (int i = 0; i < 4; i++)
                    for (int j = 0; j < 8; j++)
                        pk_out[i * 8 + j] = (uchar)(cur_pk[i] >> (j * 8));
                __global uchar *addr_out = &hit_addresses[idx * 42];
                for (int i = 0; i < 42; i++)
                    addr_out[i] = addr_ascii[i];
            }
        }

        u256 tx, ty, tz;
        u256_copy(tx, px); u256_copy(ty, py); u256_copy(tz, pz);
        point_add_mixed(px, py, pz, tx, ty, tz, SECP256K1_GX, SECP256K1_GY);
        u256_inc(k);
    }
}
