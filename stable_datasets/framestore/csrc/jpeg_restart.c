/* Lossless re-coding of JPEG frames with a restart marker after every MCU (codec.to_restart).
 *
 * restart_one / restart_many: through libjpeg (read the DCT coefficients, write them back with restart_interval
 *   set), as jpegtran does; any input libjpeg reads.
 * fast_restart_many: the fast path. Parses each frame's Huffman-coded scan itself, re-codes only the first DC
 *   coefficient of each component in each MCU (the DC predictor resets at every restart marker) and copies every
 *   other bit unchanged, writing a header identical to libjpeg's, so its output is byte-identical to restart_many,
 *   except in padding blocks (past a component's edge in the last MCU column or row): libjpeg rewrites those (no AC,
 *   the previous block's DC), this copies what the source has there. Decoders discard them, so pixels are the same;
 *   they differ only for encoders other than libjpeg (ffmpeg's fills them with image content).
 *   Frames it does not handle (progressive, non-standard Huffman tables, Adobe/RGB, multi-scan, corrupt) return
 *   status 2, and codec.to_restart sends those through libjpeg.
 *
 * Built and loaded by native.load (links libjpeg); called through ctypes, which releases the GIL.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <setjmp.h>
#include <jpeglib.h>

struct err { struct jpeg_error_mgr pub; jmp_buf jb; };
static void on_error(j_common_ptr c) { longjmp(((struct err *)c->err)->jb, 1); }
static void on_message(j_common_ptr c) { (void)c; }

/* What jpegtran -restart <interval>B does: DCT coefficients copied, re-coded with default (baseline, standard
   Huffman) parameters and a restart marker every `interval` MCUs, COM markers copied. 0 ok, 1 ok with libjpeg
   warnings (e.g. corrupt data), -1 failed. */
static int restart_one(const uint8_t *src, unsigned long n, int interval, unsigned char **out, unsigned long *out_n) {
    struct jpeg_decompress_struct d;
    struct jpeg_compress_struct c;
    struct err e;
    d.err = jpeg_std_error(&e.pub);
    c.err = &e.pub;
    e.pub.error_exit = on_error;
    e.pub.output_message = on_message;
    *out = NULL;
    *out_n = 0;
    jpeg_create_decompress(&d);
    jpeg_create_compress(&c);
    if (setjmp(e.jb)) {
        jpeg_destroy_compress(&c);
        jpeg_destroy_decompress(&d);
        free(*out);
        *out = NULL;
        return -1;
    }
    jpeg_mem_src(&d, src, n);
    jpeg_save_markers(&d, JPEG_COM, 0xFFFF);
    jpeg_read_header(&d, TRUE);
    jvirt_barray_ptr *coef = jpeg_read_coefficients(&d);
    long warnings = e.pub.num_warnings;  /* jpeg_write_coefficients resets the count */
    jpeg_copy_critical_parameters(&d, &c);
    c.restart_interval = interval;
    c.restart_in_rows = 0;
    jpeg_mem_dest(&c, out, out_n);
    jpeg_write_coefficients(&c, coef);
    for (jpeg_saved_marker_ptr m = d.marker_list; m != NULL; m = m->next)
        jpeg_write_marker(&c, m->marker, m->data, m->data_length);
    jpeg_finish_compress(&c);
    jpeg_finish_decompress(&d);
    int warned = warnings + e.pub.num_warnings > 0;
    jpeg_destroy_compress(&c);
    jpeg_destroy_decompress(&d);
    return warned;
}

/* ------------------------------------------------------------------------------------------------------------
   Fast path: the same output for baseline frames coded with the standard Huffman tables, without libjpeg's
   coefficient round trip. With a restart after every MCU, the only symbols that change are each component's first
   DC difference in an MCU (its predictor restarts at 0); every other bit is copied. Frames it does not handle
   (progressive, other tables, Adobe/RGB, multi-scan, corrupt) get status 2 and go through restart_one.
   ------------------------------------------------------------------------------------------------------------ */
typedef struct { uint8_t bits[17]; uint8_t vals[256]; int count; } dht_t;
typedef struct {
    uint16_t lut[512];      /* top 9 bits -> (code length << 8) | symbol; 0: longer code */
    uint16_t skip[2048];    /* AC: top 11 bits -> bits to skip, code and extra bits (0-4), coefficients advanced
                               (5-9), end of block (10), a run of 16 zeros (11); 0: a code longer than 11 bits */
    int32_t maxcode[18], valptr[17], mincode[17];
    uint8_t vals[256];
} dec_t;
typedef struct { uint16_t code[256]; uint8_t size[256]; } enc_t;

static dht_t STD[4];  /* DC luminance, AC luminance, DC chrominance, AC chrominance: libjpeg's own defaults */
static dec_t STD_DEC[4];
static enc_t STD_ENC[4];

static void build(const dht_t *t, dec_t *d, enc_t *e) {
    int code = 0, k = 0;
    memset(d->lut, 0, sizeof d->lut);
    memcpy(d->vals, t->vals, 256);
    for (int len = 1; len <= 16; len++) {
        d->valptr[len] = k;
        d->mincode[len] = code;
        for (int i = 0; i < t->bits[len]; i++, k++, code++) {
            uint8_t sym = t->vals[k];
            if (e) { e->code[sym] = (uint16_t)code; e->size[sym] = (uint8_t)len; }
            if (len <= 9)
                for (int j = 0; j < (1 << (9 - len)); j++) d->lut[(code << (9 - len)) | j] = (uint16_t)((len << 8) | sym);
        }
        d->maxcode[len] = t->bits[len] ? code - 1 : -1;
        code <<= 1;
    }
    d->maxcode[17] = 0x7fffffff;
    /* AC skip table: a symbol's whole length (code + its extra bits) and how far it moves through the block */
    memset(d->skip, 0, sizeof d->skip);
    code = 0; k = 0;
    for (int len = 1; len <= 16; len++, code <<= 1)
        for (int i = 0; i < t->bits[len]; i++, k++, code++) {
            int rs = t->vals[k], r = rs >> 4, sz = rs & 15;
            if (len > 11) continue;
            int adv = sz ? r + 1 : (r == 15 ? 16 : 0), eob = !sz && r != 15, zrl = !sz && r == 15;
            uint16_t e = (uint16_t)((len + sz) | (adv << 5) | (eob << 10) | (zrl << 11));
            for (int j = 0; j < (1 << (11 - len)); j++) d->skip[(code << (11 - len)) | j] = e;
        }
}

int fast_init(void) {
    struct jpeg_compress_struct c;
    struct jpeg_error_mgr e;
    c.err = jpeg_std_error(&e);
    jpeg_create_compress(&c);
    c.in_color_space = JCS_RGB;
    c.input_components = 3;
    jpeg_set_defaults(&c);
    JHUFF_TBL *t[4] = {c.dc_huff_tbl_ptrs[0], c.ac_huff_tbl_ptrs[0], c.dc_huff_tbl_ptrs[1], c.ac_huff_tbl_ptrs[1]};
    for (int i = 0; i < 4; i++) {
        memset(&STD[i], 0, sizeof STD[i]);
        for (int l = 1; l <= 16; l++) { STD[i].bits[l] = t[i]->bits[l]; STD[i].count += t[i]->bits[l]; }
        memcpy(STD[i].vals, t[i]->huffval, STD[i].count);
        build(&STD[i], &STD_DEC[i], &STD_ENC[i]);
    }
    jpeg_destroy_compress(&c);
    return 0;
}

static inline uint64_t peek(const uint8_t *c, int64_t pos) {  /* the 64 bits from bit pos (c padded by 16 bytes) */
    const uint8_t *p = c + (pos >> 3);
    uint64_t v;
    memcpy(&v, p, 8);
    v = __builtin_bswap64(v);
    int r = (int)(pos & 7);
    return r ? (v << r) | (p[8] >> (8 - r)) : v;
}

/* bit reader with the next bits in a register: bb holds nb valid bits, most significant first */
typedef struct { const uint8_t *c; int64_t bp; uint64_t bb; int nb; } rd_t;

static inline void rd_seek(rd_t *r, int64_t byte) { r->bp = byte; r->bb = 0; r->nb = 0; }
static inline void rd_fill(rd_t *r) {  /* branch-free: top up to 56..63 bits */
    uint64_t v;
    memcpy(&v, r->c + r->bp, 8);
    r->bb |= __builtin_bswap64(v) >> r->nb;
    r->bp += (63 - r->nb) >> 3;
    r->nb |= 56;
}
static inline int64_t rd_pos(const rd_t *r) { return r->bp * 8 - r->nb; }
static inline void rd_skip(rd_t *r, int n) { r->bb <<= n; r->nb -= n; }

/* A restart interval (or the scan) whose data ends at byte `end` was decoded exactly as libjpeg decodes it: no bit
   read past its end (libjpeg would read zeros there, and warn) and less than a byte left over (libjpeg warns of
   extraneous data). */
static inline int interval_end(const rd_t *r, int64_t end) {
    int64_t left = end * 8 - rd_pos(r);
    return left >= 0 && left < 8;
}

static inline int rd_huff(const dec_t *d, rd_t *r) {
    if (r->nb < 16) rd_fill(r);
    uint16_t l = d->lut[r->bb >> 55];
    if (l) { rd_skip(r, l >> 8); return l & 255; }
    uint32_t w = (uint32_t)(r->bb >> 48);
    for (int len = 10; len <= 16; len++) {
        int32_t code = (int32_t)(w >> (16 - len));
        if (code <= d->maxcode[len]) { rd_skip(r, len); return d->vals[d->valptr[len] + code - d->mincode[len]]; }
    }
    return -1;
}

static inline int huff(const dec_t *d, const uint8_t *c, int64_t *pos) {
    uint64_t v = peek(c, *pos);
    uint16_t l = d->lut[v >> 55];
    if (l) { *pos += l >> 8; return l & 255; }
    uint32_t w = (uint32_t)(v >> 48);
    for (int len = 10; len <= 16; len++) {
        int32_t code = (int32_t)(w >> (16 - len));
        if (code <= d->maxcode[len]) { *pos += len; return d->vals[d->valptr[len] + code - d->mincode[len]]; }
    }
    return -1;
}

typedef struct { uint8_t *o; int64_t n, cap; uint64_t acc; int nb; } wr_t;

static inline void put(wr_t *w, uint64_t bits, int n) {  /* n <= 32 */
    w->acc = (w->acc << n) | (bits & ((1ull << n) - 1));
    w->nb += n;
    while (w->nb >= 8) {
        uint8_t b = (uint8_t)(w->acc >> (w->nb - 8));
        w->nb -= 8;
        if (w->n + 2 <= w->cap) { w->o[w->n++] = b; if (b == 0xFF) w->o[w->n++] = 0; } else w->n = w->cap + 1;
    }
}

static inline void raw(wr_t *w, int b) { if (w->n < w->cap) w->o[w->n++] = (uint8_t)b; else w->n = w->cap + 1; }
static inline void raw16(wr_t *w, int v) { raw(w, v >> 8); raw(w, v & 255); }

static void emit_dht(wr_t *w, const dht_t *t, int cls, int idx) {
    raw(w, 0xFF); raw(w, 0xC4); raw16(w, 2 + 1 + 16 + t->count); raw(w, (cls << 4) | idx);
    for (int l = 1; l <= 16; l++) raw(w, t->bits[l]);
    for (int i = 0; i < t->count; i++) raw(w, t->vals[i]);
}

static int same_table(const dht_t *a, const dht_t *b) {
    return a->count == b->count && !memcmp(a->bits + 1, b->bits + 1, 16) && !memcmp(a->vals, b->vals, a->count);
}

#define BE16(p) (((int)(p)[0] << 8) | (p)[1])
#define FALLBACK 2

/* One frame; clean: scratch of n + 16 bytes; seg: scratch of n / 2 + 2 entries. 0 ok, FALLBACK otherwise. */
static int fast_one(const uint8_t *s, int64_t n, uint8_t *out, int64_t cap, int64_t *out_n, uint8_t *clean,
                    int64_t *seg) {
    int jfif = 0, jmaj = 1, jmin = 1, junit = 0, jx = 1, jy = 1, h = 0, w = 0, nc = 0, dri = 0, got_sof = 0;
    int cid[3], chv[3], ctq[3], ctd[3], cta[3], qprec[4] = {0}, qhave[4] = {0};
    uint16_t q[4][64];
    dht_t tab[2][4];
    int thave[2][4] = {{0}};
    const uint8_t *com[32];
    int ncom = 0, comlen[32];
    int64_t i = 2, sos_end = 0;
    if (n < 4 || s[0] != 0xFF || s[1] != 0xD8) return FALLBACK;
    while (i + 4 <= n) {
        if (s[i] != 0xFF) return FALLBACK;
        int m = s[i + 1];
        if (m == 0xFF) { i++; continue; }
        if (m == 0x01 || (m >= 0xD0 && m <= 0xD7)) { i += 2; continue; }
        /* libjpeg reads APPn, COM, DQT, DHT, DRI and SOFn before a scan and refuses other markers (JPGn, a second
           SOI, ...); SOF kinds this does not handle are sent to it below */
        if (!((m >= 0xE0 && m <= 0xEF) || m == 0xFE || m == 0xDB || m == 0xC4 || m == 0xDD || m == 0xDA ||
              (m >= 0xC0 && m <= 0xCF && m != 0xC8)))
            return FALLBACK;
        int L = BE16(s + i + 2);
        if (L < 2 || i + 2 + L > n) return FALLBACK;
        const uint8_t *p = s + i + 4, *end = s + i + 2 + L;
        if (m == 0xE0 && L >= 16 && !memcmp(p, "JFIF\0", 5)) {
            jfif = 1;
            if (p[5] != 1 && p[5] != 2) return FALLBACK;  /* libjpeg warns of other JFIF versions */
            jmaj = p[5]; jmin = p[6];
            junit = p[7]; jx = BE16(p + 8); jy = BE16(p + 10);
        } else if (m == 0xEE && L >= 14 && !memcmp(p, "Adobe", 5)) {
            return FALLBACK;
        } else if (m == 0xFE) {
            if (ncom == 32) return FALLBACK;
            com[ncom] = p; comlen[ncom++] = L - 2;
        } else if (m == 0xDB) {
            while (p < end) {
                int pq = p[0] >> 4, t = p[0] & 15;
                if (t > 3 || pq > 1 || p + 1 + 64 * (pq + 1) > end) return FALLBACK;
                for (int k = 0; k < 64; k++) q[t][k] = pq ? (uint16_t)BE16(p + 1 + 2 * k) : p[1 + k];
                qhave[t] = 1;
                qprec[t] = 0;
                for (int k = 0; k < 64; k++) if (q[t][k] > 255) qprec[t] = 1;
                p += 1 + 64 * (pq + 1);
            }
        } else if (m == 0xC4) {
            while (p < end) {
                int tc = p[0] >> 4, th = p[0] & 15;
                if (tc > 1 || th > 3 || p + 17 > end) return FALLBACK;
                dht_t *t = &tab[tc][th];
                memset(t, 0, sizeof *t);
                for (int l = 1; l <= 16; l++) { t->bits[l] = p[l]; t->count += p[l]; }
                if (t->count > 256 || p + 17 + t->count > end) return FALLBACK;
                memcpy(t->vals, p + 17, t->count);
                thave[tc][th] = 1;
                p += 17 + t->count;
            }
        } else if (m == 0xC0 || m == 0xC1) {
            if (L < 8 || p[0] != 8) return FALLBACK;
            h = BE16(p + 1); w = BE16(p + 3); nc = p[5];
            if ((nc != 1 && nc != 3) || L != 8 + 3 * nc || h == 0 || w == 0) return FALLBACK;
            for (int c = 0; c < nc; c++) { cid[c] = p[6 + 3 * c]; chv[c] = p[7 + 3 * c]; ctq[c] = p[8 + 3 * c]; }
            got_sof = 1;
        } else if (m >= 0xC2 && m <= 0xCF && m != 0xC4 && m != 0xC8 && m != 0xCC) {
            return FALLBACK;  /* progressive, lossless, hierarchical, arithmetic */
        } else if (m == 0xDD) {
            if (L != 4) return FALLBACK;
            dri = BE16(p);
        } else if (m == 0xDA) {
            if (!got_sof || p[0] != nc || L != 6 + 2 * nc) return FALLBACK;
            for (int c = 0; c < nc; c++) {
                if (p[1 + 2 * c] != cid[c]) return FALLBACK;  /* scan order must be frame order */
                ctd[c] = p[2 + 2 * c] >> 4; cta[c] = p[2 + 2 * c] & 15;
            }
            if (p[1 + 2 * nc] != 0 || p[2 + 2 * nc] != 63 || p[3 + 2 * nc] != 0) return FALLBACK;
            sos_end = i + 2 + L;
            break;
        }
        i += 2 + L;
    }
    if (!sos_end) return FALLBACK;
    if (nc == 3 && !jfif) return FALLBACK;  /* libjpeg may take it for RGB */
    /* the scan's tables must be the standard ones libjpeg writes: luminance for component 0, chrominance else */
    const dec_t *dd[3], *ad[3];
    const enc_t *de[3];
    for (int c = 0; c < nc; c++) {
        int r = c == 0 ? 0 : 2;
        if (ctd[c] > 3 || cta[c] > 3 || !thave[0][ctd[c]] || !thave[1][cta[c]] || ctq[c] > 3 || !qhave[ctq[c]])
            return FALLBACK;
        if (!same_table(&tab[0][ctd[c]], &STD[r]) || !same_table(&tab[1][cta[c]], &STD[r + 1])) return FALLBACK;
        dd[c] = &STD_DEC[r]; ad[c] = &STD_DEC[r + 1]; de[c] = &STD_ENC[r];
    }
    /* entropy-coded data, unstuffed; seg[k]: where the k-th restart interval after the first starts */
    int64_t cn = 0, ns = 0, j = sos_end;
    int eoi = 0;
    while (j < n) {
        const uint8_t *ff = memchr(s + j, 0xFF, n - j);  /* copy up to the next 0xFF in one go */
        int64_t run = ff ? ff - (s + j) : n - j;
        memcpy(clean + cn, s + j, run);
        cn += run;
        j += run;
        if (j >= n) break;
        if (j + 1 >= n) break;
        uint8_t x = s[j + 1];
        if (x == 0x00) { clean[cn++] = 0xFF; j += 2; }
        else if (x >= 0xD0 && x <= 0xD7) {  /* libjpeg resyncs (and warns) on a marker out of order */
            if (x - 0xD0 != (ns & 7)) return FALLBACK;
            seg[ns++] = cn; j += 2;
        }
        else if (x == 0xFF) j++;
        else if (x == 0xD9) { eoi = 1; break; }
        else return FALLBACK;  /* DNL, another scan, ... */
    }
    if (!eoi) return FALLBACK;
    memset(clean + cn, 0, 16);
    /* MCU geometry */
    int hmax = 1, vmax = 1, nblk[3];
    for (int c = 0; c < nc; c++) {
        int hc = chv[c] >> 4, vc = chv[c] & 15;
        if (hc < 1 || hc > 4 || vc < 1 || vc > 4) return FALLBACK;
        if (hc > hmax) hmax = hc;
        if (vc > vmax) vmax = vc;
        nblk[c] = hc * vc;
    }
    if (nc == 1) { hmax = vmax = 1; nblk[0] = 1; }
    int64_t mx = (w + 8 * hmax - 1) / (8 * hmax), my = (h + 8 * vmax - 1) / (8 * vmax), M = mx * my;
    if (ns != (dri ? (M - 1) / dri : 0)) return FALLBACK;  /* a marker where libjpeg ends the data, or one missing */
    /* header, as libjpeg writes it */
    wr_t W = {out, 0, cap, 0, 0};
    raw(&W, 0xFF); raw(&W, 0xD8);
    raw(&W, 0xFF); raw(&W, 0xE0); raw16(&W, 16);
    raw(&W, 'J'); raw(&W, 'F'); raw(&W, 'I'); raw(&W, 'F'); raw(&W, 0);
    raw(&W, jmaj); raw(&W, jmin); raw(&W, junit); raw16(&W, jx); raw16(&W, jy); raw(&W, 0); raw(&W, 0);
    for (int k = 0; k < ncom; k++) {
        raw(&W, 0xFF); raw(&W, 0xFE); raw16(&W, comlen[k] + 2);
        for (int b = 0; b < comlen[k]; b++) raw(&W, com[k][b]);
    }
    int sent[4] = {0}, prec = 0;
    for (int c = 0; c < nc; c++) {
        int t = ctq[c];
        if (sent[t]) continue;
        sent[t] = 1;
        prec |= qprec[t];
        raw(&W, 0xFF); raw(&W, 0xDB); raw16(&W, 2 + 1 + 64 * (qprec[t] + 1)); raw(&W, (qprec[t] << 4) | t);
        for (int k = 0; k < 64; k++) { if (qprec[t]) raw(&W, q[t][k] >> 8); raw(&W, q[t][k] & 255); }
    }
    raw(&W, 0xFF); raw(&W, prec ? 0xC1 : 0xC0); raw16(&W, 8 + 3 * nc); raw(&W, 8); raw16(&W, h); raw16(&W, w);
    raw(&W, nc);
    for (int c = 0; c < nc; c++) { raw(&W, cid[c]); raw(&W, chv[c]); raw(&W, ctq[c]); }
    emit_dht(&W, &STD[0], 0, 0);
    emit_dht(&W, &STD[1], 1, 0);
    if (nc == 3) { emit_dht(&W, &STD[2], 0, 1); emit_dht(&W, &STD[3], 1, 1); }
    raw(&W, 0xFF); raw(&W, 0xDD); raw16(&W, 4); raw16(&W, 1);
    raw(&W, 0xFF); raw(&W, 0xDA); raw16(&W, 6 + 2 * nc); raw(&W, nc);
    for (int c = 0; c < nc; c++) { raw(&W, cid[c]); raw(&W, c == 0 ? 0x00 : 0x11); }
    raw(&W, 0); raw(&W, 63); raw(&W, 0);
    /* MCUs */
    int64_t limit = cn * 8 + 64, next_seg = 0;
    int pred[3] = {0, 0, 0};
    rd_t R;
    rd_seek(&R, 0);
    R.c = clean;
    for (int64_t mcu = 0; mcu < M; mcu++) {
        if (dri && mcu > 0 && mcu % dri == 0) {
            if (next_seg >= ns || !interval_end(&R, seg[next_seg])) return FALLBACK;
            rd_seek(&R, seg[next_seg++]);
            pred[0] = pred[1] = pred[2] = 0;
        }
        for (int c = 0; c < nc; c++) {
            int64_t copy_from = 0;
            for (int b = 0; b < nblk[c]; b++) {
                int t = rd_huff(dd[c], &R);
                if (t < 0 || t > 11) return FALLBACK;
                int diff = 0;
                if (t) {
                    if (R.nb < 16) rd_fill(&R);
                    diff = (int)(R.bb >> (64 - t));
                    rd_skip(&R, t);
                    if (diff < (1 << (t - 1))) diff -= (1 << t) - 1;
                }
                pred[c] += diff;
                if (b == 0) {  /* the predictor restarts at 0: code the DC value itself */
                    int v = pred[c], a = v < 0 ? -v : v, cat = 0;
                    while (a) { cat++; a >>= 1; }
                    if (cat > 11) return FALLBACK;  /* |DC| <= 2047 here, so the at most 15 more differences of
                                                       an MCU (each <= 2047) cannot wrap libjpeg's 16-bit values */
                    put(&W, de[c]->code[cat], de[c]->size[cat]);
                    if (cat) put(&W, (uint64_t)(v < 0 ? v - 1 : v), cat);
                    copy_from = rd_pos(&R);
                }
                /* Only the symbols' lengths matter: the bits are copied. That gives libjpeg's bytes only for the
                   symbols its encoder would write for these coefficients: no coefficient past the 64th, and no
                   run of 16 zeros that ends the block (it writes the end of block instead). */
                int zrl = 0, k = 1;  /* the loop stops once past the 64th: one check after it covers all */
                while (k < 64) {
                    if (R.nb < 32) rd_fill(&R);  /* a code (<= 16) and its extra bits (<= 10) */
                    uint16_t e = ad[c]->skip[R.bb >> 53];
                    if (e) {
                        rd_skip(&R, e & 31);
                        if (e & (1 << 10)) break;
                        zrl = e >> 11;
                        k += (e >> 5) & 31;
                        continue;
                    }
                    int rs = rd_huff(ad[c], &R);
                    if (rs < 0) return FALLBACK;
                    int r = rs >> 4, sz = rs & 15;
                    if (sz == 0) {
                        if (r != 15) break;
                        zrl = 1;
                        k += 16;
                        continue;
                    }
                    zrl = 0;
                    k += r + 1;
                    if (R.nb < 16) rd_fill(&R);
                    rd_skip(&R, sz);
                }
                if (zrl | (k > 64)) return FALLBACK;
                if (R.bp > cn + 8) return FALLBACK;
            }
            int64_t pos = rd_pos(&R);
            if (pos > limit) return FALLBACK;
            for (int64_t a = copy_from; a < pos;) {  /* the rest of this component's blocks, bit for bit */
                int k = pos - a > 32 ? 32 : (int)(pos - a);
                put(&W, peek(clean, a) >> (64 - k), k);
                a += k;
            }
        }
        if (W.nb) put(&W, (1ull << (8 - W.nb)) - 1, 8 - W.nb);  /* pad with 1 bits to a byte */
        raw(&W, 0xFF);
        raw(&W, mcu == M - 1 ? 0xD9 : 0xD0 + (int)(mcu & 7));
    }
    if (!interval_end(&R, cn)) return FALLBACK;
    if (W.n > cap) return -3;  /* out of space: the caller grows the buffer */
    *out_n = W.n;
    return 0;
}

/* As restart_many (interval 1 only), with the fast path where it applies; status 2 = not handled here. */
int64_t fast_restart_many(const uint8_t *buf, const int64_t *off, int64_t n, uint8_t *out, int64_t cap,
                          int64_t *out_off, int32_t *status) {
    int64_t big = 0, o = 0;
    for (int64_t i = 0; i < n; i++) if (off[i + 1] - off[i] > big) big = off[i + 1] - off[i];
    uint8_t *clean = malloc(big + 32);
    int64_t *seg = malloc(sizeof(int64_t) * (big / 2 + 2));
    out_off[0] = 0;
    for (int64_t i = 0; i < n; i++) {
        int64_t sz = 0;
        int r = fast_one(buf + off[i], off[i + 1] - off[i], out + o, cap - o, &sz, clean, seg);
        if (r == -3) { free(clean); free(seg); return -1 - i; }
        status[i] = r;
        if (r == 0) o += sz;
        out_off[i + 1] = o;
    }
    free(clean);
    free(seg);
    return o;
}

/* Frames buf[off[i]:off[i + 1]], i < n, re-coded back to back into out (capacity cap); out_off[n + 1]; status[i]
   from restart_one (a failed frame gets length 0). Returns bytes written, or -1 - i if frame i did not fit. */
int64_t restart_many(const uint8_t *buf, const int64_t *off, int64_t n, int32_t interval, uint8_t *out, int64_t cap,
                     int64_t *out_off, int32_t *status) {
    int64_t o = 0;
    out_off[0] = 0;
    for (int64_t i = 0; i < n; i++) {
        unsigned char *p;
        unsigned long sz;
        status[i] = restart_one(buf + off[i], (unsigned long)(off[i + 1] - off[i]), interval, &p, &sz);
        if (status[i] >= 0) {
            if (o + (int64_t)sz > cap) {
                free(p);
                return -1 - i;
            }
            memcpy(out + o, p, sz);
            o += (int64_t)sz;
            free(p);
        }
        out_off[i + 1] = o;
    }
    return o;
}
