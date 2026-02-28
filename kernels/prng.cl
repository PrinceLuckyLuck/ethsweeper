/*
 * xoshiro256** PRNG for OpenCL
 *
 * State: 4 x ulong (256 bits)
 * Output: ulong (64 bits)
 * Period: 2^256 - 1
 *
 * Reference: https://prng.di.unimi.it/xoshiro256starstar.c
 */

typedef struct {
    ulong s0;
    ulong s1;
    ulong s2;
    ulong s3;
} xoshiro256ss_state;

inline ulong rotl(const ulong x, int k) {
    return (x << k) | (x >> (64 - k));
}

inline ulong xoshiro256ss_next(xoshiro256ss_state *state) {
    const ulong result = rotl(state->s1 * 5, 7) * 9;
    const ulong t = state->s1 << 17;

    state->s2 ^= state->s0;
    state->s3 ^= state->s1;
    state->s1 ^= state->s2;
    state->s0 ^= state->s3;

    state->s2 ^= t;
    state->s3 = rotl(state->s3, 45);

    return result;
}

/*
 * Initialize PRNG state from a 32-byte seed buffer.
 * seed points to 4 consecutive ulongs (little-endian).
 */
inline void xoshiro256ss_init(xoshiro256ss_state *state, __global const ulong *seed) {
    state->s0 = seed[0];
    state->s1 = seed[1];
    state->s2 = seed[2];
    state->s3 = seed[3];
    // Ensure state is not all-zero
    if (state->s0 == 0 && state->s1 == 0 && state->s2 == 0 && state->s3 == 0) {
        state->s0 = 1;
    }
}

/*
 * Generate a 256-bit (32-byte) private key from PRNG.
 * Writes 4 ulongs = 32 bytes into privkey[0..3].
 */
inline void generate_privkey(xoshiro256ss_state *state, ulong privkey[4]) {
    privkey[0] = xoshiro256ss_next(state);
    privkey[1] = xoshiro256ss_next(state);
    privkey[2] = xoshiro256ss_next(state);
    privkey[3] = xoshiro256ss_next(state);
}
