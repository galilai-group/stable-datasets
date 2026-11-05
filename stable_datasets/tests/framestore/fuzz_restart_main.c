/* Memory-error check of framestore/csrc/jpeg_restart.c under AddressSanitizer and UBSan: re-codes every file given (or listed in
   a directory) with the fast translator and with libjpeg, and checks that their statuses agree (see fuzz_memory.sh). */
#include "../../framestore/csrc/jpeg_restart.c"
#include <dirent.h>

static int one(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *s = malloc(n ? n : 1);  /* exactly n bytes: ASan sees any read past the frame */
    if (fread(s, 1, n, f) != (size_t)n) { fclose(f); free(s); return 0; }
    fclose(f);
    int64_t off[2] = {0, n}, cap = 3 * n + 8192, fo[2], so[2];
    int32_t fs, ss;
    uint8_t *a = malloc(cap), *b = malloc(cap);
    int64_t r1 = fast_restart_many(s, off, 1, a, cap, fo, &fs);
    int64_t r2 = restart_many(s, off, 1, 1, b, cap, so, &ss);
    int bad = 0;
    if (r1 >= 0 && r2 >= 0 && fs != 2 && fs != ss) {  /* bytes may differ in padding blocks: the pytest checks */
        printf("DIFFERS %s: fast status %d (%lld bytes), libjpeg %d (%lld bytes)\n", path, fs, (long long)fo[1], ss,
               (long long)so[1]);
        bad = 1;
    }
    free(s); free(a); free(b);
    return bad;
}

int main(int argc, char **argv) {
    fast_init();
    int files = 0, bad = 0;
    for (int i = 1; i < argc; i++) {
        DIR *d = opendir(argv[i]);
        if (!d) { files++; bad += one(argv[i]); continue; }
        char p[4096];
        for (struct dirent *e; (e = readdir(d));)
            if (e->d_name[0] != '.') { snprintf(p, sizeof p, "%s/%s", argv[i], e->d_name); files++; bad += one(p); }
        closedir(d);
    }
    printf("%d frames, %d differ\n", files, bad);
    return bad != 0;
}
