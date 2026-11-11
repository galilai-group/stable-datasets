/* Many preads per call (pread_many.PreadMany), so a thread issues its share of a batch's reads without the GIL.
 * Built and loaded by native.load; called through ctypes.
 */
#include <stdint.h>
#include <unistd.h>

/* The hash of this file a prebuilt copy (a wheel's) was built from: _native.load uses that copy only if it matches */
#ifndef FRAMESTORE_KEY
#define FRAMESTORE_KEY ""
#endif
const char *framestore_key(void) { return FRAMESTORE_KEY; }
int64_t pread_many(const int64_t *fd, const int64_t *src, const int64_t *dst, const int64_t *n, int64_t count,
                   uint8_t *buf) {
    int64_t total = 0;
    for (int64_t i = 0; i < count; i++) {
        int64_t done = 0;
        while (done < n[i]) {
            ssize_t r = pread((int)fd[i], buf + dst[i] + done, (size_t)(n[i] - done), (off_t)(src[i] + done));
            if (r <= 0) return -1 - i;
            done += r;
        }
        total += done;
    }
    return total;
}
