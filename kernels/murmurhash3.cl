/*
 * MurmurHash3_x86_32 for OpenCL
 *
 * Port of the reference C implementation.
 * Used for Bloom filter hashing (7 seeds: 0..6).
 *
 * Reference: https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
 */

inline uint rotl32(uint x, int r) {
    return (x << r) | (x >> (32 - r));
}

inline uint fmix32(uint h) {
    h ^= h >> 16;
    h *= 0x85ebca6bU;
    h ^= h >> 13;
    h *= 0xc2b2ae35U;
    h ^= h >> 16;
    return h;
}

/*
 * MurmurHash3_x86_32
 *
 * key:  pointer to data bytes
 * len:  length of data in bytes
 * seed: hash seed
 *
 * Returns: 32-bit unsigned hash value
 */
inline uint murmurhash3_32(const uchar *key, int len, uint seed) {
    const int nblocks = len / 4;
    uint h1 = seed;

    const uint c1 = 0xcc9e2d51U;
    const uint c2 = 0x1b873593U;

    // Body - process 4-byte blocks
    for (int i = 0; i < nblocks; i++) {
        uint k1 = ((uint)key[i * 4 + 0])
                 | ((uint)key[i * 4 + 1] << 8)
                 | ((uint)key[i * 4 + 2] << 16)
                 | ((uint)key[i * 4 + 3] << 24);

        k1 *= c1;
        k1 = rotl32(k1, 15);
        k1 *= c2;

        h1 ^= k1;
        h1 = rotl32(h1, 13);
        h1 = h1 * 5 + 0xe6546b64U;
    }

    // Tail - remaining bytes
    const uchar *tail = key + nblocks * 4;
    uint k1 = 0;

    switch (len & 3) {
        case 3: k1 ^= ((uint)tail[2]) << 16; // fall through
        case 2: k1 ^= ((uint)tail[1]) << 8;  // fall through
        case 1: k1 ^= ((uint)tail[0]);
                k1 *= c1;
                k1 = rotl32(k1, 15);
                k1 *= c2;
                h1 ^= k1;
    }

    // Finalization
    h1 ^= (uint)len;
    h1 = fmix32(h1);

    return h1;
}
