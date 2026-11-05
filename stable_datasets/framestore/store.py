"""A frame store opened for reading: one JPEG per row in a lance dataset (with episode_idx, h, w and, once converted,
seg_len), its frames located by their blob descriptors and read without copies.

    store = FrameStore("/vol/frames.lance", index="shared", read="mmap")
    rows = store.windows(rng, clips=64, frames=16, stride=2)   # random clips: rows of 16 frames, every 2nd row
    fid, pos, size, h, w = store.descriptors(rows)             # where each frame's bytes are (store.bufs[fid][pos:...])
    off, h, w = store.copy_frames(rows, out)                   # whole frames back to back into out

Read policies (read= or $FRAMESTORE_READ), measured on an H100 node with the token loader (README):
    mmap    memory-map the data files (default): best when the store is in the page cache
    pread   one exact-size pread per frame from a thread pool: local NVMe that the page cache does not hold
    spans   one read per clip's stretch of a file (~2x the bytes, 1/16 the reads): NFS and other storage limited by
            reads per second
    ranges  only the byte ranges a batch needs (token loader); loses to spans on NFS
    take    a store whose frame column is not blob-encoded: lance take per batch (slow: ~100 us per frame)
"""

import mmap
import os

import numpy as np


try:
    import numba
except ImportError:
    numba = None

if numba is not None:

    @numba.njit(cache=True, nogil=True)
    def _copy_frames(bufs, fid, pos, size, off, out):
        for f in range(len(pos)):
            out[off[f] : off[f + 1]] = bufs[fid[f]][pos[f] : pos[f] + size[f]]


def _readonly(a):
    a = a.view(np.ndarray)  # plain read-only ndarrays, so numba sees one array type for every buffer
    a.setflags(write=False)
    return a


class FrameStore:
    """A blob-encoded frame store (convert.py) opened for zero-copy reads: frame bytes come from np.memmap of the
    lance files, located by the blob descriptors (kind, position, size, blob_id). Every fragment has one data file
    holding its frames under 64 KB (kind 0); larger frames live in its side files data/<data file>/<blob_id>.blob
    (kind 1). Each file is memory-mapped once; a frame is addressed as (file id, position).

    index=True reads every row's descriptor (and seg_len) once into numpy, so a batch touches lance not at all (for
    stores whose index fits in every worker's memory); index="shared" memory-maps the node's one copy of the same
    (index.py: built once into /dev/shm by the first process); index=False takes them from lance per batch. For
    cold-storage runs, FRAMESTORE_MADV_RANDOM=1 turns off the maps' fault readahead, and drop_pages() unmaps what a
    batch touched.

    read="pread" (or FRAMESTORE_READ=pread) reads each batch's frames with pread instead, from a pool of threads
    (FRAMESTORE_READ_THREADS), into one buffer that stands in for the maps: one exact-size read per frame, where a page
    fault on a map reads its 4 KB page plus readahead and blocks the worker meanwhile (on storage the page cache does
    not hold, the maps read ~5x a frame's bytes per clip, or one synchronous 4 KB read per page with MADV_RANDOM).
    read="ranges" reads only byte ranges a caller asks for (fetch_ranges; the token loader reads each clip's first
    header and the needed MCUs); FRAMESTORE_ODIRECT=1 opens the files O_DIRECT (NFS: byte-exact reads that
    bypass the page cache; local filesystems need aligned O_DIRECT reads, which this does not do). read="spans" reads
    each clip's frames in one read (a clip's frames are neighbouring rows, so one stretch of one file, frames over
    64 KB aside): ~2x the bytes of its frames in 1/frames the reads, for storage limited by reads per second (NFS).
    Stores without seg_len (not converted) serve whole frames only. A store whose frame column is not blob-encoded
    is read by lance take instead: each batch's bytes are taken into one arrow buffer, which stands in for the
    memory maps."""

    def __init__(self, path, index=False, read=None, column="frame", episode_column="episode_idx"):
        import lance
        import pyarrow as pa
        from numba.typed import List

        self.ds = lance.dataset(path)
        # a blob column's take returns (kind, position, size, ...) descriptors instead of the bytes
        self.column = column
        self.blob = pa.types.is_struct(self.ds.take([0], columns=[column]).schema.field(column).type)
        self.has_seg = "seg_len" in self.ds.schema.names
        ep = self.ds.to_table(columns=[episode_column]).column(episode_column).to_numpy()
        cut = np.flatnonzero(np.diff(ep)) + 1
        self.ep_start = np.concatenate(([0], cut))
        self.ep_len = np.diff(np.concatenate((self.ep_start, [len(ep)])))
        self.index = None
        self.List = List
        self.maps = []
        self.read = (read or os.environ.get("FRAMESTORE_READ", "mmap")) if self.blob else "take"
        self.fds, self.pool = [], None
        self.frames_per_clip = 16  # rows are clips of this many frames, for read="spans" (windows() sets it)
        if not self.blob:  # lance take per batch
            return
        from .index import FrameIndex, store_files

        frags = self.ds.get_fragments()
        self.frag_start = np.cumsum([0] + [f.count_rows() for f in frags])
        files, self.data_fid, self.side_fid = store_files(path, frags)
        self.bufs = List([self._map(f) for f in files])
        if index == "shared":  # one per node, memory-mapped by every worker (index.py)
            self.index = FrameIndex.shared(path)
            assert self.index.files == files, "index built for other files"
        elif index:
            self.index = self._columns(self.ds.take(np.arange(len(ep)), columns=self._cols()), np.arange(len(ep)))

    def _map(self, f):
        if self.read in ("pread", "ranges", "spans"):
            direct = os.O_DIRECT if os.environ.get("FRAMESTORE_ODIRECT") == "1" else 0
            self.fds.append(os.open(f, os.O_RDONLY | direct))
            return np.empty(0, np.uint8)  # bufs are the batch's buffer instead (_fetch)
        m = np.memmap(f, np.uint8, mode="r")
        if os.environ.get("FRAMESTORE_MADV_RANDOM") == "1":
            m._mmap.madvise(mmap.MADV_RANDOM)
        self.maps.append(m._mmap)
        return _readonly(m)

    def drop_pages(self):
        """Unmap every page this process has touched, so posix_fadvise(DONTNEED) can evict them (strictly cold runs)."""
        for m in self.maps:
            m.madvise(mmap.MADV_DONTNEED)

    def _cols(self):
        return [self.column, "h", "w"] + (["seg_len"] if self.has_seg else [])

    def _columns(self, t, rows):
        d = t.column(self.column).combine_chunks()
        if self.blob:
            kind = d.field("kind").to_numpy()
            frag = np.searchsorted(self.frag_start, rows, side="right") - 1
            fid = np.where(kind == 0, self.data_fid[frag], self.side_fid[frag, d.field("blob_id").to_numpy()])
            assert kind.max() <= 1 and fid.min() >= 0, "unexpected blob descriptor"
            out = {
                "fid": fid,
                "pos": d.field("position").to_numpy().astype(np.int64),
                "size": d.field("size").to_numpy().astype(np.int64),
            }
        else:  # the taken bytes themselves: one arrow data buffer, frames at their offsets (no per-frame copies)
            import pyarrow as pa

            _, off, buf = d.buffers()
            width = np.int64 if pa.types.is_large_binary(d.type) else np.int32  # binary: 32-bit offsets
            o = np.frombuffer(off, width)[d.offset : d.offset + len(d) + 1].astype(np.int64)
            self.bufs = self.List([np.frombuffer(buf, np.uint8)])
            out = {"fid": np.zeros(len(d), np.int64), "pos": o[:-1].copy(), "size": np.diff(o)}
        out.update(h=t.column("h").to_numpy(), w=t.column("w").to_numpy())
        if "seg_len" in t.column_names:
            sl = t.column("seg_len").combine_chunks()
            _, off, buf = sl.buffers()
            o = np.frombuffer(off, np.int32)[sl.offset : sl.offset + len(sl) + 1].astype(np.int64)
            out["seg"] = np.frombuffer(buf, np.uint8)[o[0] : o[-1]].copy().view(np.uint16)
            out["seg_off"] = (o - o[0]) // 2
        return out

    def windows(self, rng, clips, frames=16, stride=1):
        """Rows of `clips` random clips (clip-major, flattened): `frames` rows each, every `stride`-th row of one
        episode; an episode is drawn uniformly among those long enough, then the clip's start uniformly in it."""
        self.frames_per_clip = frames
        span = (frames - 1) * stride + 1
        ok = np.flatnonzero(self.ep_len >= span)
        e = ok[rng.integers(0, len(ok), clips)]
        first = self.ep_start[e] + (rng.random(clips) * (self.ep_len[e] - span + 1)).astype(np.int64)
        return (first[:, None] + stride * np.arange(frames)[None]).reshape(-1)

    def descriptors(self, rows):
        """file id, position, size, h, w per row."""
        if self.index is None:
            c = self._columns(self.ds.take(rows, columns=[self.column, "h", "w"]), rows)
        elif not isinstance(self.index, dict):
            c = self.index.lookup(rows)
        else:
            c = {k: self.index[k][rows] for k in ("fid", "pos", "size", "h", "w")}
        fid, pos = self._fetch(c["fid"], c["pos"], c["size"])
        return fid, pos, c["size"], c["h"], c["w"]

    def take(self, rows):
        """file id, pos, seg_len (concatenated), segments per row, h, w."""
        c = self.take_meta(rows)
        fid, pos = self._fetch(c["fid"], c["pos"], c["size"])
        return fid, pos, c["seg"], c["nseg"], c["h"], c["w"]

    def take_meta(self, rows):
        """Descriptors (fid, pos, size, h, w) and seg_len (seg, nseg) of rows, without reading any frame."""
        if self.index is not None and not isinstance(self.index, dict):
            return self.index.lookup(rows)
        if self.index is None:
            c = self._columns(self.ds.take(rows, columns=self._cols()), rows)
            seg, nseg = c["seg"], np.diff(c["seg_off"])
        else:
            a, b = self.index["seg_off"][rows], self.index["seg_off"][rows + 1]
            c = {k: self.index[k][rows] for k in ("fid", "pos", "size", "h", "w")}
            seg, nseg = _gather_ranges(self.index["seg"], a, b), b - a
        c.update(seg=seg, nseg=nseg)
        return c

    def _reader(self):
        if self.pool is None:
            from .preads import PreadMany

            self.pool = PreadMany(int(os.environ.get("FRAMESTORE_READ_THREADS", "16")))
        return self.pool

    def fetch_spans(self, fid, pos, size, limit=8 << 20):
        """One read per (clip, file): from its first frame's start to its last frame's end (frames are rows in
        order, so a clip's frames in one file lie in one stretch of it); stretches over `limit` bytes are read frame
        by frame. Returns the frames' (file id, position) in one batch buffer, as _fetch."""
        clip = np.arange(len(fid)) // self.frames_per_clip
        key = clip * (int(fid.max()) + 1) + fid
        u, inv = np.unique(key, return_inverse=True)
        lo = np.full(len(u), np.iinfo(np.int64).max, np.int64)
        hi = np.zeros(len(u), np.int64)
        np.minimum.at(lo, inv, pos)
        np.maximum.at(hi, inv, pos + size)
        big = (hi - lo) > limit
        if big.any():  # unexpected layout: those groups frame by frame
            one = big[inv]
            lo_f, hi_f = pos.copy(), pos + size
            lo = np.concatenate((lo[~big], lo_f[one]))
            hi = np.concatenate((hi[~big], hi_f[one]))
            gfid = np.concatenate((u[~big] % (int(fid.max()) + 1), fid[one]))
            ginv = np.where(one, len(u) - big.sum() + np.cumsum(one) - 1, np.cumsum(~big)[inv] - 1)
        else:
            gfid, ginv = u % (int(fid.max()) + 1), inv
        n = hi - lo
        off = np.concatenate(([0], np.cumsum(n)))
        if getattr(self, "_rbuf", None) is None or len(self._rbuf) < off[-1]:
            self._rbuf = np.empty(int(off[-1] * 1.25), np.uint8)
        self._reader()(np.asarray(self.fds)[gfid], lo, off[:-1], n, self._rbuf)
        self.bufs = self.List([self._rbuf])
        return np.zeros(len(fid), np.int64), off[ginv] + pos - lo[ginv]

    def fetch_ranges(self, fid, pos, size, fr, a, b):
        """Byte ranges [a, b) of frames fr into a batch buffer that holds every frame at its own offset (the rest
        of each frame left unread); the frames' (file id, position) in it, as _fetch."""
        off = np.concatenate(([0], np.cumsum(size)))
        if getattr(self, "_rbuf", None) is None or len(self._rbuf) < off[-1]:
            self._rbuf = np.empty(int(off[-1] * 1.25), np.uint8)
        self._reader()(np.asarray(self.fds)[fid[fr]], pos[fr] + a, off[fr] + a, b - a, self._rbuf)
        self.bufs = self.List([self._rbuf])
        return np.zeros(len(fid), np.int64), off[:-1]

    def _fetch(self, fid, pos, size):
        """read="pread": the frames' bytes into one buffer (self.bufs), by a thread pool; their (file id, position)
        in it. Otherwise the frames stay where they are."""
        if self.read == "spans":
            return self.fetch_spans(fid, pos, size)
        if self.read not in ("pread", "ranges"):
            return fid, pos
        return self.fetch_ranges(fid, pos, size, np.arange(len(fid)), np.zeros(len(fid), np.int64), size)

    def copy_frames(self, rows, out):
        """Whole frames' bytes, back to back, into out; returns offsets (len(rows) + 1). ValueError if they do not
        fit."""
        fid, pos, size, h, w = self.descriptors(rows)
        off = np.concatenate(([0], np.cumsum(size)))
        if off[-1] > len(out):
            raise ValueError(f"{len(rows)} frames take {off[-1]:,} bytes; the buffer holds {len(out):,}")
        _copy_frames(self.bufs, fid, pos, size, off, out)
        return off, h, w

    def frame(self, fid, pos, size):
        """One frame's bytes (a view of the memory map)."""
        return self.bufs[fid][pos : pos + size]


def _gather_ranges(x, a, b):
    idx = np.repeat(a - np.cumsum(np.concatenate(([0], (b - a)[:-1]))), b - a) + np.arange(int((b - a).sum()))
    return x[idx]
