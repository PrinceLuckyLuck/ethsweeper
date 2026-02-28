/*
 * Keccak-256 for OpenCL
 *
 * Ethereum variant: padding 0x01 (NOT SHA3 padding 0x06)
 * Input: 64 bytes (uncompressed public key without 04 prefix)
 * Output: 32 bytes hash (last 20 bytes = Ethereum address)
 *
 * Keccak state: 5x5 matrix of 64-bit words = 200 bytes
 * Rate for Keccak-256: 1088 bits = 136 bytes
 * Capacity: 512 bits = 64 bytes
 */

// Round constants
__constant ulong KECCAK_RC[24] = {
    0x0000000000000001UL, 0x0000000000008082UL, 0x800000000000808AUL,
    0x8000000080008000UL, 0x000000000000808BUL, 0x0000000080000001UL,
    0x8000000080008081UL, 0x8000000000008009UL, 0x000000000000008AUL,
    0x0000000000000088UL, 0x0000000080008009UL, 0x000000008000000AUL,
    0x000000008000808BUL, 0x800000000000008BUL, 0x8000000000008089UL,
    0x8000000000008003UL, 0x8000000000008002UL, 0x8000000000000080UL,
    0x000000000000800AUL, 0x800000008000000AUL, 0x8000000080008081UL,
    0x8000000000008080UL, 0x0000000080000001UL, 0x8000000080008008UL
};

// Rotation offsets
__constant int KECCAK_ROTC[24] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
    27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44
};

// Pi permutation indices
__constant int KECCAK_PILN[24] = {
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1
};

inline ulong keccak_rotl64(ulong x, int n) {
    return (x << n) | (x >> (64 - n));
}

inline void keccak_f1600(ulong state[25]) {
    ulong bc[5], t;

    for (int round = 0; round < 24; round++) {
        // Theta
        for (int i = 0; i < 5; i++)
            bc[i] = state[i] ^ state[i + 5] ^ state[i + 10] ^ state[i + 15] ^ state[i + 20];

        for (int i = 0; i < 5; i++) {
            t = bc[(i + 4) % 5] ^ keccak_rotl64(bc[(i + 1) % 5], 1);
            for (int j = 0; j < 25; j += 5)
                state[j + i] ^= t;
        }

        // Rho and Pi
        t = state[1];
        for (int i = 0; i < 24; i++) {
            int j = KECCAK_PILN[i];
            bc[0] = state[j];
            state[j] = keccak_rotl64(t, KECCAK_ROTC[i]);
            t = bc[0];
        }

        // Chi
        for (int j = 0; j < 25; j += 5) {
            for (int i = 0; i < 5; i++)
                bc[i] = state[j + i];
            for (int i = 0; i < 5; i++)
                state[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
        }

        // Iota
        state[0] ^= KECCAK_RC[round];
    }
}

/*
 * Keccak-256 hash of exactly 64 bytes input.
 * Optimized for the Ethereum pubkey->address use case.
 *
 * input:  64 bytes (public key X || Y)
 * output: 32 bytes (hash)
 */
inline void keccak256_64(const uchar input[64], uchar output[32]) {
    ulong state[25];

    // Initialize state to zero
    for (int i = 0; i < 25; i++) state[i] = 0;

    // Absorb: 64 bytes = 8 ulongs (rate = 136 bytes = 17 ulongs, so fits in one block)
    // XOR input into state (little-endian)
    for (int i = 0; i < 8; i++) {
        ulong v = 0;
        for (int j = 0; j < 8; j++) {
            v |= (ulong)input[i * 8 + j] << (j * 8);
        }
        state[i] = v;
    }

    // Padding: Ethereum Keccak uses 0x01 padding (not SHA3 0x06)
    // Byte 64 gets 0x01, byte 135 (last byte of rate) gets 0x80
    // state[8] ^= 0x01 (byte 64 = first byte of word 8)
    state[8] ^= 0x01UL;
    // state[16] ^= 0x80 << 56 (byte 135 = last byte of word 16)
    state[16] ^= 0x8000000000000000UL;

    // Permutation
    keccak_f1600(state);

    // Squeeze: output 32 bytes = 4 ulongs (little-endian)
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            output[i * 8 + j] = (uchar)(state[i] >> (j * 8));
        }
    }
}

/*
 * Extract Ethereum address (last 20 bytes of keccak256) and convert to
 * ASCII hex string "0x" + 40 hex chars = 42 bytes.
 */
inline void hash_to_eth_address(const uchar hash[32], uchar addr_ascii[42]) {
    const char hex_chars[] = "0123456789abcdef";
    addr_ascii[0] = '0';
    addr_ascii[1] = 'x';
    // Address = last 20 bytes of hash (bytes 12..31)
    for (int i = 0; i < 20; i++) {
        uchar b = hash[12 + i];
        addr_ascii[2 + i * 2] = hex_chars[(b >> 4) & 0x0F];
        addr_ascii[2 + i * 2 + 1] = hex_chars[b & 0x0F];
    }
}
