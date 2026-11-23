import numpy as np
import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("numba")

from stable_datasets.codecs import Chunks, RestartJpeg  # noqa: E402
from stable_datasets.codecs.chunks import encode  # noqa: E402
from stable_datasets.loading import (  # noqa: E402
    Planner,
    Records,
    SamplingIndex,
    Uniform,
    fetch_and_decode,
)
from stable_datasets.sources import LocalSource  # noqa: E402
from stable_datasets.transforms import Chain, Flip, Normalize, RandomCrop, RandomResizedCrop  # noqa: E402


class CountingSource(LocalSource):
    def read_into(self, fid, offset, size, buf, dst):
        self.reads = getattr(self, "reads", 0) + len(fid)
        self.bytes = getattr(self, "bytes", 0) + int(np.sum(size))
        return super().read_into(fid, offset, size, buf, dst)


@pytest.fixture(scope="module")
def jpeg_store(tmp_path_factory):
    from stable_datasets.framestore import codec as fcodec
    from stable_datasets.tests.framestore.test_gpu_jpeg import frames, restart_coded

    imgs = frames(n=3, h=136, w=200) + frames(n=2, h=77, w=133, seed=4)
    coded = restart_coded(imgs[:3], quality=90, subsampling=2) + restart_coded(imgs[3:], quality=80, subsampling=0)
    path = tmp_path_factory.mktemp("jpegs") / "frames.bin"
    pad = 1000  # bytes between records, as in a real file
    blob = b"".join(b"\0" * pad + j.tobytes() for j, _ in coded)
    path.write_bytes(blob)
    sizes = [len(j) for j, _ in coded]
    pos = np.cumsum([pad] + [s + pad for s in sizes[:-1]])
    buf = np.concatenate([j for j, _ in coded])
    off = np.concatenate(([0], np.cumsum(sizes)))
    info = fcodec.probe(buf, off)
    seg, seg_off, ok = fcodec.segment_lengths(buf, off, info)
    assert ok.all()
    meta = {k: info[k] for k in ("h", "w", "mcu_h", "mcu_w")}
    meta["sos_end"] = info["sos_end"] - off[:-1]
    records = Records([str(path)], np.zeros(len(coded), np.int64), pos, meta, seg, seg_off, {})
    full = [torch.from_numpy(RestartJpeg.decode_region(j, s, (0, 0, 10**6, 10**6))).permute(2, 0, 1) for j, s in coded]
    return records, full


CHAINS = [
    Chain(RandomResizedCrop(48, scale=(0.05, 0.3)), Flip(), Normalize()),
    Chain(RandomCrop((40, 64))),
    Chain(Flip(p=1.0)),
]


@pytest.mark.parametrize("chain", CHAINS, ids=["rrc", "crop", "whole"])
def test_plan_execution_equals_the_chain_on_whole_images(jpeg_store, chain):
    records, full = jpeg_store
    planner = Planner(RestartJpeg(), records, chain)
    plan = planner.plan(np.random.default_rng(0), 12)
    got = fetch_and_decode(plan, planner, LocalSource(records.files))
    for i, r in enumerate(plan.record):
        want = chain.apply(full[r], plan.params, i)
        assert torch.equal(got[i], want), i


def test_small_crops_read_a_fraction_of_the_bytes(jpeg_store):
    records, _ = jpeg_store
    # gap 16: MCU segments 2 bytes apart (their restart markers) merge into one read
    planner = Planner(RestartJpeg(), records, Chain(RandomResizedCrop(32, scale=(0.01, 0.03))), gap=16)
    plan = planner.plan(np.random.default_rng(1), 64)
    source = CountingSource(records.files)
    fetch_and_decode(plan, planner, source)
    whole = sum(int(records.meta["sos_end"][r] + records.ragged[records.ragged_off[r] : records.ragged_off[r + 1]].sum())
                for r in plan.record)  # fmt: skip
    assert source.bytes < 0.5 * whole
    assert source.reads < len(plan.piece_unit)


def test_a_larger_gap_gives_fewer_reads(jpeg_store):
    records, _ = jpeg_store
    chain = Chain(RandomResizedCrop(32, scale=(0.05, 0.2)))
    a = Planner(RestartJpeg(), records, chain, gap=0).plan(np.random.default_rng(2), 32)
    b = Planner(RestartJpeg(), records, chain, gap=1 << 20).plan(np.random.default_rng(2), 32)
    assert len(b.reads.fid) <= len(a.reads.fid) and len(b.reads.fid) == 1
    assert np.array_equal(a.boxes, b.boxes)


def test_sampling_boxes_confine_the_chain(jpeg_store):
    records, full = jpeg_store
    index = SamplingIndex(np.array([0, 1, 3]), box=np.array([[8, 16, 72, 96], [0, 0, 40, 40], [20, 30, 77, 133]]))
    chain = Chain(RandomCrop((24, 24)))
    planner = Planner(RestartJpeg(), records, chain, index=index, sampler=Uniform())
    plan = planner.plan(np.random.default_rng(3), 20)
    got = fetch_and_decode(plan, planner, LocalSource(records.files))
    for i in range(len(plan)):
        r, (y0, x0, y1, x1) = plan.record[i], index.box[plan.rows[i]]
        assert (plan.boxes[i, :2] >= [y0, x0]).all() and (plan.boxes[i, 2:] <= [y1, x1]).all()
        assert torch.equal(got[i], chain.apply(full[r][:, y0:y1, x0:x1], plan.params, i))


def test_chunked_volumes(tmp_path):
    rng = np.random.default_rng(0)
    vols = [rng.integers(-1000, 1000, s).astype(np.int16) for s in [(40, 50, 30), (33, 64, 64), (20, 20, 20)]]
    path = tmp_path / "vols.bin"
    blobs, sizes, shapes, pos, at = [], [], [], [], 0
    for v in vols:
        bl, meta = encode(v, (16, 16, 16), compression="zlib")
        pos.append(at)
        at += sum(len(b) for b in bl)
        blobs += bl
        sizes.append([len(b) for b in bl])
        shapes.append(meta["shape"])
    path.write_bytes(b"".join(blobs))
    ragged = np.concatenate([np.array(s) for s in sizes])
    off = np.concatenate(([0], np.cumsum([len(s) for s in sizes])))
    m = {"shape": np.array(shapes), "chunk": np.full((3, 3), 16)}
    records = Records([str(path)], np.zeros(3, np.int64), np.array(pos), m, ragged, off,
                      {"dtype": "<i2", "compression": "zlib"})  # fmt: skip
    chain = Chain(RandomCrop((16, 24, 16)))
    planner = Planner(Chunks(), records, chain)
    plan = planner.plan(np.random.default_rng(1), 10)
    got = fetch_and_decode(plan, planner, LocalSource(records.files))
    for i, r in enumerate(plan.record):
        assert torch.equal(got[i], chain.apply(torch.from_numpy(vols[r])[None], plan.params, i))
