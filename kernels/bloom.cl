/*
 * Bloom filter check on GPU
 *
 * Uses MurmurHash3 with k=7 seeds (0..6).
 * Bloom filter data is stored in global memory (VRAM).
 *
 * Parameters:
 *   bloom_data: pointer to bloom filter bit array in VRAM
 *   addr_bytes: ASCII bytes of the address ("0x" + 40 hex chars = 42 bytes)
 *   addr_len:   length of address (42)
 *   bloom_m:    total number of bits in bloom filter
 *   bloom_k:    number of hash functions (7)
 *
 * Returns: 1 if possibly present, 0 if definitely absent
 */

inline int bloom_check(__global const uchar *bloom_data,
                       const uchar *addr_bytes,
                       int addr_len,
                       ulong bloom_m,
                       int bloom_k) {
    for (int seed = 0; seed < bloom_k; seed++) {
        uint h = murmurhash3_32(addr_bytes, addr_len, (uint)seed);
        ulong bit_pos = (ulong)h % bloom_m;
        ulong byte_idx = bit_pos >> 3;
        int bit_idx = (int)(bit_pos & 7);
        if (!(bloom_data[byte_idx] & (1 << bit_idx))) {
            return 0;
        }
    }
    return 1;
}
