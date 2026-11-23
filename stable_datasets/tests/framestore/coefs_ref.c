/* The reference for stable_datasets.framestore.gpu_jpeg: libjpeg's own entropy decode of a baseline JPEG (jpeg_read_coefficients),
 * as the quantized DCT coefficients of every block in MCU order -- for each MCU row, MCU column, component, block row
 * within the MCU, block column: 64 coefficients in natural (not zigzag) order. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <setjmp.h>
#include <jpeglib.h>

const char *framestore_key(void) { return ""; }

struct err { struct jpeg_error_mgr pub; jmp_buf jb; };
static void on_error(j_common_ptr c) { longjmp(((struct err *)c->err)->jb, 1); }
static void on_message(j_common_ptr c) { (void)c; }

/* out: cap int16s. Returns the number of blocks written, -1 on a libjpeg error, -2 if out is too small. */
int64_t read_coefs(const uint8_t *jpg, int64_t n, int16_t *out, int64_t cap) {
    struct jpeg_decompress_struct d;
    struct err e;
    d.err = jpeg_std_error(&e.pub);
    e.pub.error_exit = on_error;
    e.pub.output_message = on_message;
    if (setjmp(e.jb)) { jpeg_destroy_decompress(&d); return -1; }
    jpeg_create_decompress(&d);
    jpeg_mem_src(&d, jpg, (unsigned long)n);
    jpeg_read_header(&d, TRUE);
    jvirt_barray_ptr *coef = jpeg_read_coefficients(&d);
    int64_t k = 0;
    for (JDIMENSION my = 0; my < d.MCU_rows_in_scan; my++)
        for (JDIMENSION mx = 0; mx < d.MCUs_per_row; mx++)
            for (int c = 0; c < d.num_components; c++) {
                jpeg_component_info *ci = &d.comp_info[c];
                int hs = ci->h_samp_factor, vs = ci->v_samp_factor;
                JBLOCKARRAY rows = d.mem->access_virt_barray((j_common_ptr)&d, coef[c], my * vs, vs, FALSE);
                for (int v = 0; v < vs; v++)
                    for (int h = 0; h < hs; h++) {
                        if ((k + 1) * 64 > cap) { jpeg_destroy_decompress(&d); return -2; }
                        JCOEF *b = rows[v][mx * hs + h];
                        for (int i = 0; i < 64; i++) out[k * 64 + i] = (int16_t)b[i];
                        k++;
                    }
            }
    jpeg_finish_decompress(&d);
    jpeg_destroy_decompress(&d);
    return k;
}
