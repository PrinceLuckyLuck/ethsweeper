/*
 * secp256k1 Elliptic Curve for OpenCL
 *
 * 256-bit modular arithmetic and scalar multiplication.
 * Computes: privkey (32 bytes) -> uncompressed pubkey (64 bytes, without 04 prefix)
 *
 * Curve parameters:
 *   p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
 *   a = 0, b = 7
 *   G = (0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
 *        0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8)
 *   n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
 *
 * Representation: 256-bit integers as 8 x uint (little-endian limbs, 32 bits each)
 * u256[0] = least significant, u256[7] = most significant
 *
 * Point representation: flat arrays px[8], py[8], pz[8] (Jacobian coords)
 * to avoid OpenCL address space issues with structs.
 */

typedef uint u256[8];

// --- Constants ---

__constant uint SECP256K1_P[8] = {
    0xFFFFFC2Fu, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
    0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
};

__constant uint SECP256K1_GX[8] = {
    0x16F81798u, 0x59F2815Bu, 0x2DCE28D9u, 0x029BFCDBu,
    0xCE870B07u, 0x55A06295u, 0xF9DCBBACu, 0x79BE667Eu
};

__constant uint SECP256K1_GY[8] = {
    0xFB10D4B8u, 0x9C47D08Fu, 0xA6855419u, 0xFD17B448u,
    0x0E1108A8u, 0x5DA4FBFCu, 0x26A3C465u, 0x483ADA77u
};

// --- 256-bit Arithmetic ---

inline void u256_copy(u256 dst, const u256 src) {
    for (int i = 0; i < 8; i++) dst[i] = src[i];
}

inline void u256_zero(u256 a) {
    for (int i = 0; i < 8; i++) a[i] = 0;
}

inline int u256_is_zero(const u256 a) {
    uint r = 0;
    for (int i = 0; i < 8; i++) r |= a[i];
    return r == 0;
}

inline int u256_gte(const u256 a, __constant const uint *b) {
    for (int i = 7; i >= 0; i--) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return 0;
    }
    return 1; // equal
}

// a = a + b, returns carry
inline uint u256_add(u256 a, const u256 b) {
    ulong carry = 0;
    for (int i = 0; i < 8; i++) {
        carry += (ulong)a[i] + (ulong)b[i];
        a[i] = (uint)carry;
        carry >>= 32;
    }
    return (uint)carry;
}

inline uint u256_add_c(u256 a, __constant const uint *b) {
    ulong carry = 0;
    for (int i = 0; i < 8; i++) {
        carry += (ulong)a[i] + (ulong)b[i];
        a[i] = (uint)carry;
        carry >>= 32;
    }
    return (uint)carry;
}

// a = a - b, returns borrow
inline uint u256_sub(u256 a, const u256 b) {
    long borrow = 0;
    for (int i = 0; i < 8; i++) {
        borrow += (long)(ulong)a[i] - (long)(ulong)b[i];
        a[i] = (uint)borrow;
        borrow >>= 32;
    }
    return (uint)(borrow & 1);
}

inline uint u256_sub_c(u256 a, __constant const uint *b) {
    long borrow = 0;
    for (int i = 0; i < 8; i++) {
        borrow += (long)(ulong)a[i] - (long)(ulong)b[i];
        a[i] = (uint)borrow;
        borrow >>= 32;
    }
    return (uint)(borrow & 1);
}

// --- Modular Arithmetic (mod p) ---

inline void mod_add(u256 r, const u256 a, const u256 b) {
    u256_copy(r, a);
    uint carry = u256_add(r, b);
    if (carry || u256_gte(r, SECP256K1_P)) {
        u256_sub_c(r, SECP256K1_P);
    }
}

inline void mod_sub(u256 r, const u256 a, const u256 b) {
    u256_copy(r, a);
    uint borrow = u256_sub(r, b);
    if (borrow) {
        u256_add_c(r, SECP256K1_P);
    }
}

// r = (a * b) mod p, using secp256k1 fast reduction
// p = 2^256 - 0x1000003D1, where 0x1000003D1 = 2^32 + 977
// So 2^256 ≡ 2^32 + 977 (mod p)
// For high limb prod[8+i], contribution = prod[8+i] * (2^32 + 977) at position 32*i
//   = prod[8+i] * 977 at position 32*i (add to limb i)
//   + prod[8+i]       at position 32*(i+1) (add to limb i+1)
inline void mod_mul(u256 r, const u256 a, const u256 b) {
    uint prod[16];
    for (int i = 0; i < 16; i++) prod[i] = 0;

    // Schoolbook 8x8 multiplication
    for (int i = 0; i < 8; i++) {
        ulong carry = 0;
        for (int j = 0; j < 8; j++) {
            carry += (ulong)prod[i + j] + (ulong)a[i] * (ulong)b[j];
            prod[i + j] = (uint)carry;
            carry >>= 32;
        }
        prod[i + 8] += (uint)carry;
    }

    // Single-pass reduction of high part into low part
    // At position i, we accumulate:
    //   prod[i]             (low part)
    //   prod[i+8] * 977     (*977 contribution)
    //   prod[i+7] if i>0    (*2^32 contribution from previous high limb)
    ulong carry = 0;
    carry = (ulong)prod[0] + (ulong)prod[8] * 977UL;
    r[0] = (uint)carry;
    carry >>= 32;
    for (int i = 1; i < 8; i++) {
        carry += (ulong)prod[i] + (ulong)prod[i + 8] * 977UL + (ulong)prod[i + 7];
        r[i] = (uint)carry;
        carry >>= 32;
    }
    carry += (ulong)prod[15]; // *2^32 part of last high limb -> overflow

    // Reduce overflow: carry * 2^256 ≡ carry * (2^32 + 977) (mod p)
    // carry is small (< ~2^35), so carry * (2^32 + 977) fits in ulong
    while (carry > 0) {
        ulong c2 = 0;
        ulong lo = carry * 977UL;
        uint hi_add = (uint)carry; // carry * 2^32 -> add to limb 1

        c2 = (ulong)r[0] + (uint)lo;
        r[0] = (uint)c2;
        c2 >>= 32;
        c2 += (ulong)r[1] + (lo >> 32) + (ulong)hi_add;
        r[1] = (uint)c2;
        c2 >>= 32;
        for (int i = 2; i < 8; i++) {
            c2 += (ulong)r[i];
            r[i] = (uint)c2;
            c2 >>= 32;
        }
        carry = c2;
    }

    if (u256_gte(r, SECP256K1_P)) u256_sub_c(r, SECP256K1_P);
    if (u256_gte(r, SECP256K1_P)) u256_sub_c(r, SECP256K1_P);
}

inline void mod_sqr(u256 r, const u256 a) {
    mod_mul(r, a, a);
}

// Modular inverse: a^(p-2) mod p (Fermat's little theorem)
inline void mod_inv(u256 r, const u256 a) {
    u256 x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x223;

    mod_sqr(x2, a);     mod_mul(x2, x2, a);
    mod_sqr(x3, x2);    mod_mul(x3, x3, a);

    u256_copy(x6, x3);
    for (int i = 0; i < 3; i++) mod_sqr(x6, x6);
    mod_mul(x6, x6, x3);

    u256_copy(x9, x6);
    for (int i = 0; i < 3; i++) mod_sqr(x9, x9);
    mod_mul(x9, x9, x3);

    u256_copy(x11, x9);
    for (int i = 0; i < 2; i++) mod_sqr(x11, x11);
    mod_mul(x11, x11, x2);

    u256_copy(x22, x11);
    for (int i = 0; i < 11; i++) mod_sqr(x22, x22);
    mod_mul(x22, x22, x11);

    u256_copy(x44, x22);
    for (int i = 0; i < 22; i++) mod_sqr(x44, x44);
    mod_mul(x44, x44, x22);

    u256_copy(x88, x44);
    for (int i = 0; i < 44; i++) mod_sqr(x88, x88);
    mod_mul(x88, x88, x44);

    u256_copy(x176, x88);
    for (int i = 0; i < 88; i++) mod_sqr(x176, x176);
    mod_mul(x176, x176, x88);

    u256_copy(x220, x176);
    for (int i = 0; i < 44; i++) mod_sqr(x220, x220);
    mod_mul(x220, x220, x44);

    u256_copy(x223, x220);
    for (int i = 0; i < 3; i++) mod_sqr(x223, x223);
    mod_mul(x223, x223, x3);

    // p-2 = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D
    // Binary from MSB: [223 ones] [0] [22 ones] [0000] [1] [0] [1] [1] [0] [1]
    u256_copy(r, x223);
    mod_sqr(r, r); // bit 32 = 0
    for (int i = 0; i < 22; i++) { mod_sqr(r, r); mod_mul(r, r, a); } // 22 ones
    mod_sqr(r, r); // 0
    mod_sqr(r, r); // 0
    mod_sqr(r, r); // 0
    mod_sqr(r, r); // 0
    mod_sqr(r, r); mod_mul(r, r, a); // 1
    mod_sqr(r, r); // 0
    mod_sqr(r, r); mod_mul(r, r, a); // 1
    mod_sqr(r, r); mod_mul(r, r, a); // 1
    mod_sqr(r, r); // 0
    mod_sqr(r, r); mod_mul(r, r, a); // 1
}

// --- Point operations (Jacobian coordinates, flat arrays) ---
// Point = (px[8], py[8], pz[8])
// Infinity: pz == 0

inline void point_set_infinity(u256 px, u256 py, u256 pz) {
    u256_zero(px);
    u256_zero(py);
    py[0] = 1;
    u256_zero(pz);
}

// Point doubling: (rx,ry,rz) = 2*(px,py,pz)
// a = 0 for secp256k1
inline void point_double(u256 rx, u256 ry, u256 rz,
                         const u256 px, const u256 py, const u256 pz) {
    if (u256_is_zero(pz)) {
        point_set_infinity(rx, ry, rz);
        return;
    }

    u256 A, B, C, D, E, F, t;

    mod_sqr(A, px);       // A = X1^2
    mod_sqr(B, py);       // B = Y1^2
    mod_sqr(C, B);        // C = B^2

    // D = 2*((X1+B)^2 - A - C)
    mod_add(D, px, B);
    mod_sqr(D, D);
    mod_sub(D, D, A);
    mod_sub(D, D, C);
    mod_add(D, D, D);

    // E = 3*A
    mod_add(E, A, A);
    mod_add(E, E, A);

    mod_sqr(F, E);        // F = E^2

    // X3 = F - 2*D
    mod_add(t, D, D);
    mod_sub(rx, F, t);

    // Y3 = E*(D - X3) - 8*C
    mod_sub(t, D, rx);
    mod_mul(ry, E, t);
    mod_add(C, C, C); mod_add(C, C, C); mod_add(C, C, C); // 8C
    mod_sub(ry, ry, C);

    // Z3 = 2*Y1*Z1
    mod_mul(rz, py, pz);
    mod_add(rz, rz, rz);
}

// Mixed point addition: (rx,ry,rz) = (px,py,pz) + (qx,qy,1)
// qx/qy from __constant memory (generator G)
inline void point_add_mixed(u256 rx, u256 ry, u256 rz,
                            const u256 px, const u256 py, const u256 pz,
                            __constant const uint *qx, __constant const uint *qy) {
    if (u256_is_zero(pz)) {
        for (int i = 0; i < 8; i++) rx[i] = qx[i];
        for (int i = 0; i < 8; i++) ry[i] = qy[i];
        u256_zero(rz);
        rz[0] = 1;
        return;
    }

    u256 Z1Z1, U2, S2, H, HH, I, J, rr, V, t;
    u256 qx_l, qy_l;

    for (int i = 0; i < 8; i++) qx_l[i] = qx[i];
    for (int i = 0; i < 8; i++) qy_l[i] = qy[i];

    mod_sqr(Z1Z1, pz);
    mod_mul(U2, qx_l, Z1Z1);

    mod_mul(S2, pz, Z1Z1);
    mod_mul(S2, qy_l, S2);

    mod_sub(H, U2, px);

    if (u256_is_zero(H)) {
        u256 diff;
        mod_sub(diff, S2, py);
        if (u256_is_zero(diff)) {
            point_double(rx, ry, rz, px, py, pz);
            return;
        }
        point_set_infinity(rx, ry, rz);
        return;
    }

    mod_sqr(HH, H);
    mod_add(I, HH, HH);
    mod_add(I, I, I);

    mod_mul(J, H, I);

    mod_sub(rr, S2, py);
    mod_add(rr, rr, rr);

    mod_mul(V, px, I);

    // X3 = rr^2 - J - 2*V
    mod_sqr(rx, rr);
    mod_sub(rx, rx, J);
    mod_add(t, V, V);
    mod_sub(rx, rx, t);

    // Y3 = rr*(V - X3) - 2*Y1*J
    mod_sub(t, V, rx);
    mod_mul(ry, rr, t);
    mod_mul(t, py, J);
    mod_add(t, t, t);
    mod_sub(ry, ry, t);

    // Z3 = 2*Z1*H
    mod_mul(rz, pz, H);
    mod_add(rz, rz, rz);
}

// Scalar multiplication: k * G
inline void scalar_mult_G(u256 rx, u256 ry, u256 rz, const u256 k) {
    point_set_infinity(rx, ry, rz);
    u256 tx, ty, tz;

    for (int word = 7; word >= 0; word--) {
        uint w = k[word];
        for (int bit = 31; bit >= 0; bit--) {
            // Double
            u256_copy(tx, rx); u256_copy(ty, ry); u256_copy(tz, rz);
            point_double(rx, ry, rz, tx, ty, tz);

            // Add G if bit set
            if ((w >> bit) & 1) {
                u256_copy(tx, rx); u256_copy(ty, ry); u256_copy(tz, rz);
                point_add_mixed(rx, ry, rz, tx, ty, tz, SECP256K1_GX, SECP256K1_GY);
            }
        }
    }
}

// Convert Jacobian to affine and write pubkey (64 bytes, big-endian)
inline void jacobian_to_pubkey(const u256 px, const u256 py, const u256 pz,
                               uchar pubkey_out[64]) {
    u256 z_inv, z_inv2, z_inv3, ax, ay;

    mod_inv(z_inv, pz);
    mod_sqr(z_inv2, z_inv);
    mod_mul(z_inv3, z_inv2, z_inv);

    mod_mul(ax, px, z_inv2);
    mod_mul(ay, py, z_inv3);

    for (int i = 7; i >= 0; i--) {
        int off = (7 - i) * 4;
        pubkey_out[off]     = (uchar)(ax[i] >> 24);
        pubkey_out[off + 1] = (uchar)(ax[i] >> 16);
        pubkey_out[off + 2] = (uchar)(ax[i] >> 8);
        pubkey_out[off + 3] = (uchar)(ax[i]);
    }
    for (int i = 7; i >= 0; i--) {
        int off = 32 + (7 - i) * 4;
        pubkey_out[off]     = (uchar)(ay[i] >> 24);
        pubkey_out[off + 1] = (uchar)(ay[i] >> 16);
        pubkey_out[off + 2] = (uchar)(ay[i] >> 8);
        pubkey_out[off + 3] = (uchar)(ay[i]);
    }
}

// Convert Jacobian to affine pubkey using pre-computed z_inv
inline void jacobian_to_pubkey_zi(const u256 px, const u256 py, const u256 z_inv,
                                   uchar pubkey_out[64]) {
    u256 z_inv2, z_inv3, ax, ay;

    mod_sqr(z_inv2, z_inv);
    mod_mul(z_inv3, z_inv2, z_inv);

    mod_mul(ax, px, z_inv2);
    mod_mul(ay, py, z_inv3);

    for (int i = 7; i >= 0; i--) {
        int off = (7 - i) * 4;
        pubkey_out[off]     = (uchar)(ax[i] >> 24);
        pubkey_out[off + 1] = (uchar)(ax[i] >> 16);
        pubkey_out[off + 2] = (uchar)(ax[i] >> 8);
        pubkey_out[off + 3] = (uchar)(ax[i]);
    }
    for (int i = 7; i >= 0; i--) {
        int off = 32 + (7 - i) * 4;
        pubkey_out[off]     = (uchar)(ay[i] >> 24);
        pubkey_out[off + 1] = (uchar)(ay[i] >> 16);
        pubkey_out[off + 2] = (uchar)(ay[i] >> 8);
        pubkey_out[off + 3] = (uchar)(ay[i]);
    }
}

// Montgomery batch inversion for 4 values: z0,z1,z2,z3 inverted in-place
// Cost: 1 mod_inv + 9 mod_mul (vs 4 mod_inv = ~1200 mod_mul)
inline void batch_mod_inv_4(u256 z0, u256 z1, u256 z2, u256 z3) {
    u256 p1, p2, p3;
    mod_mul(p1, z0, z1);           // p1 = z0*z1
    mod_mul(p2, p1, z2);           // p2 = z0*z1*z2
    mod_mul(p3, p2, z3);           // p3 = z0*z1*z2*z3

    u256 c;
    mod_inv(c, p3);                // c = (z0*z1*z2*z3)^(-1)

    u256 tmp;
    // z3_inv = c * p2, then c = c * z3_old
    u256_copy(tmp, z3);
    mod_mul(z3, c, p2);            // z3 = z3^(-1)
    mod_mul(c, c, tmp);            // c = (z0*z1*z2)^(-1)

    // z2_inv = c * p1, then c = c * z2_old
    u256_copy(tmp, z2);
    mod_mul(z2, c, p1);            // z2 = z2^(-1)
    mod_mul(c, c, tmp);            // c = (z0*z1)^(-1)

    // z1_inv = c * z0, then c = c * z1_old
    u256_copy(tmp, z1);
    mod_mul(z1, c, z0);            // z1 = z1^(-1)
    mod_mul(c, c, tmp);            // c = z0^(-1)

    u256_copy(z0, c);              // z0 = z0^(-1)
}

// Increment u256 by 1
inline void u256_inc(u256 a) {
    for (int i = 0; i < 8; i++) {
        a[i]++;
        if (a[i] != 0) break;  // no carry
    }
}

// Convert u256 (8 x uint, little-endian limbs) to 4 ulongs (little-endian)
inline void u256_to_privkey(const u256 k, ulong out[4]) {
    out[0] = (ulong)k[0] | ((ulong)k[1] << 32);
    out[1] = (ulong)k[2] | ((ulong)k[3] << 32);
    out[2] = (ulong)k[4] | ((ulong)k[5] << 32);
    out[3] = (ulong)k[6] | ((ulong)k[7] << 32);
}

// Main: privkey (4 ulongs LE) -> pubkey (64 bytes BE)
inline void privkey_to_pubkey(const ulong privkey[4], uchar pubkey[64]) {
    u256 k;
    k[0] = (uint)(privkey[0] & 0xFFFFFFFF);
    k[1] = (uint)(privkey[0] >> 32);
    k[2] = (uint)(privkey[1] & 0xFFFFFFFF);
    k[3] = (uint)(privkey[1] >> 32);
    k[4] = (uint)(privkey[2] & 0xFFFFFFFF);
    k[5] = (uint)(privkey[2] >> 32);
    k[6] = (uint)(privkey[3] & 0xFFFFFFFF);
    k[7] = (uint)(privkey[3] >> 32);

    u256 rx, ry, rz;
    scalar_mult_G(rx, ry, rz, k);
    jacobian_to_pubkey(rx, ry, rz, pubkey);
}
