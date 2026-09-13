# Benchmark Decision Log

Design decisions for the ICLR benchmark grid, with the reasoning and evidence
behind each. Newest section last. Entries marked **OPEN** are not yet settled.

Measurements quoted here come from the `samibg/finalized-stable-datasets` W&B
project: per-epoch times are the median wall-clock gap between consecutive epoch
starts, taken from run history (not `runtime / epochs`, which under-reports
badly on resumed runs — one cell showed 399 epochs in 342 s because the runtime
only covered the resumed tail). Estimates are from 647 main-grid ViT-S/16 runs
and 205 ResNet-50 runs; a rank-1 (dataset x method) reconstruction reproduces
the grid total within 1.1%.

---

## 0. W&B project: `stable-datasets-iclr`

New runs log to **`samibg/stable-datasets-iclr`** (`wandb.project`, overridable
via `$SDS_WANDB_PROJECT`). The previous project, `finalized-stable-datasets`,
holds the older runs — including the post-2026-07 contaminated ones and the
curated pre-contamination originals the recovery scripts read. Keeping the ICLR
sweep in its own project means telling them apart never depends on filtering by
date or tag.

**The benchmark restarts from scratch here.** Every script in the tree — including
the ones written for the old online-probe recovery effort
(`scrape_wandb_online.py`, `recover_online_probe.py`,
`merge_recovered_online_probe.py`, `resume_progress_report.py`) — now points at
the new project and the new root. Nothing reads `finalized-stable-datasets` or
`~/scratch/.stable-datasets` any more; that data is retained but orphaned.

Overrides, for pointing any of it elsewhere without editing code:

| env var | default |
| --- | --- |
| `SDS_WANDB_PROJECT` | `stable-datasets-iclr` |
| `SDS_WANDB_ENTITY_PROJECT` | `samibg/stable-datasets-iclr` |
| `STABLE_DATASETS_ROOT` | `~/scratch/stable-datasets-iclr` |
| `SDS_PROBE_PROTOCOL` | `common` |

One consequence worth knowing: `recover_online_probe.py` previously pinned the
legacy flat-`nn.Linear` probe head, because every checkpoint it recovered
predated the BN+Linear switch. Starting fresh, new checkpoints carry the BN head,
so its default is now `common`. If you ever point it at a pre-2026-09 checkpoint,
set `SDS_PROBE_PROTOCOL=legacy_linear` — `strict_loading = False` means a
mismatch does not raise, it silently reports garbage.

## 1. Methods: 7 total

`supervised, simclr, dino, mae, lejepa, nnclr, barlow_twins`.

Six SSL methods plus a supervised baseline. There is no eighth method; earlier
counts of "8" were miscounts. Verified against `launch.sh` defaults, the
`conf/model/` directory, and a sweep of all 3,604 W&B runs.

## 2. Backbones: ViT-S/16 and ResNet-50 — except MAE

Both backbones for all 20 datasets and all methods, with one exception:

**MAE is ViT-only.** It reconstructs masked *patch tokens*, which a ResNet
encoder does not produce. `run.py::_assert_supported_combo` now rejects
`mae + resnet50` at startup. Without that guard the combination launches, burns
a GPU slot, and dies hours later inside the decoder with an opaque shape error —
or worse, submitit swallows it and the job reports success with no metrics.

So the grid is 7 methods x 20 datasets on ViT-S, and 6 x 20 on ResNet-50.

ResNet-50 is **~11% slower per epoch** than ViT-S/16, measured over 137 matched
(dataset, method) cells — not faster, as one might assume from its lower nominal
FLOPs (~4.1 vs ~4.6 GFLOPs). ResNet-50's BatchNorm and memory-bound layers give
it worse arithmetic intensity, while ViT-S is nearly pure GEMM and exploits
tensor cores better under `16-mixed`. The penalty is larger on small-native-
resolution datasets (1.15x for <=33px) than on large ones (1.04x for >224px).

## 3. Datasets: 20

```
imagenet100  imagenette  cifar10  cifar100  stl10  svhn  food101  country211
cub200  fgvcaircraft  flowers102  dtd  galaxy10  pathmnist  octmnist
tissuemnist  bloodmnist  dermamnist  organamnist  pneumoniamnist
```

Culled from 42. Two criteria drove the cut:

**Redundancy.** The EMNIST family was 6 of 42 datasets but **50.5% of all
compute**, and all six are the same source corpus (NIST SD19) at different label
granularities. `emnist_byclass` and `emnist_bymerge` are the *identical* 697,932
images — bymerge only merges case-ambiguous letter classes — so that pair alone
was 36% of the grid for one dataset counted twice. All EMNIST variants are out.

**Citability.** Remaining datasets are ones with a published epoch budget we can
cite (see #4), so the training length is not an arbitrary choice of ours.

Known consequence, accepted: the cull is biased toward removing low-class-count
datasets (dropped median 9 classes vs kept 29), and the 11-30 class band is now
thin. Recorded here so it is not rediscovered as a surprise in review.

## 4. Epoch budgets — citation-pinned

Single source of truth: `benchmark_epochs` in `conf/config.yaml`, applied in
`run.py::_resolve_params` *after* the per-model `params:` blocks so all methods
share one budget per dataset. An explicit CLI `training.max_epochs=N` still wins.
Datasets absent from the table fall back to the model's own params block.

| Dataset | Base epochs | MAE epochs | Basis |
| --- | ---: | ---: | --- |
| imagenet100 | 400 | 1,600 | LeJEPA, Balestriero & LeCun 2025 (arXiv 2511.08544v3) |
| imagenette | 800 | 3,200 | Lightly SSL Imagenette benchmark |
| cifar10 | 400 | 1,600 | LeJEPA 2025 |
| cifar100 | 400 | 1,600 | LeJEPA 2025 |
| stl10 | 1,000 | 4,000 | Contrastive Learning with Synthetic Positives 2024 (arXiv 2408.16965v2) |
| svhn | 400 | 1,600 | CoViews, Bendib et al. 2024 (arXiv 2406.12847) |
| food101 | 400 | 1,600 | LeJEPA 2025 |
| country211 | 1,000 | 4,000 | **Our choice**, see note below |
| cub200 | 800 | 3,200 | Parametric Contrastive Learning 2021 (arXiv 2103.13559) |
| fgvcaircraft | 200 | 800 | Self-Supervised Fine-Grained Image Classification (ACM 3503161.3547909) |
| flowers102 | 400 | 1,600 | LeJEPA 2025 |
| dtd | 1,000 | 4,000 | GPS-SSL, Feizi et al. 2024 (arXiv 2401.01990) |
| galaxy10 | 400 | 1,600 | LeJEPA 2025 |
| pathmnist | 400 | 1,600 | Sharma et al. 2026 (arXiv 2604.01947) |
| octmnist | 400 | 1,600 | Sharma et al. 2026 |
| tissuemnist | 400 | 1,600 | Sharma et al. 2026 |
| bloodmnist | 400 | 1,600 | Sharma et al. 2026 |
| dermamnist | 400 | 1,600 | Sharma et al. 2026 |
| organamnist | 400 | 1,600 | Sharma et al. 2026 |
| pneumoniamnist | 400 | 1,600 | Sharma et al. 2026 |

**country211 is not a reproduced count.** It is our choice of a high-epoch
linear-probe budget (1,000), matching the CLIP linear-probe setting (Radford et
al. 2021) that the dataset was introduced in. Flagged as such in the config so
it is not later cited as someone else's protocol.

## 5. MAE gets 4x the epoch count

MAE masks 75% of patches, so a single MAE epoch reconstructs from a quarter of
the signal a contrastive epoch sees. Implemented as
`benchmark_epochs.mae_multiplier: 4`.

Cost consequence: MAE is the cheapest method per epoch (0.75x supervised) but
the 4x multiplier makes it the third most expensive line item overall.

## 6. Why DINO and LeJEPA dominate the budget

Both use `multicrop` with `num_global: 2, num_local: 6` — **8 views per sample**,
against 2 for SimCLR/NNCLR/BarlowTwins and 1 for supervised/MAE. In 224px-
equivalent pixels that is 2 + 6*(96/224)^2 ~= 3.1x supervised, which matches the
measured 3.33x (DINO) and 3.26x (LeJEPA) almost exactly.

Eight views at 224px also OOMs a 24 GB card at batch 256, so both run
`batch_size: 32, accumulate_grad_batches: 8` — same effective batch, 8x the
optimizer steps and kernel launches per epoch. This is inherent to multi-crop,
not a misconfiguration.

## 7. Linear probe: one common protocol for all methods

**Headline result uses an identical probe everywhere:**

| Backbone | Representation |
| --- | --- |
| ViT-S | final-layer CLS token |
| ResNet-50 | final global-average-pooled feature |

```python
probe = nn.Sequential(
    nn.BatchNorm1d(embed_dim, affine=False, eps=1e-6),
    nn.Linear(embed_dim, num_classes),
)
```

The methods' *published* probes genuinely differ:

| Method | Native representation | Native probe normalization |
| --- | --- | --- |
| MAE | final CLS token | non-affine BatchNorm |
| LeJEPA | concat of last **two** CLS tokens | LayerNorm (BN also reported) |
| DINO | concat of last **four** CLS tokens | none |
| BarlowTwins / SimCLR / NNCLR | pooled feature | none |

Adopting each method's native probe would give DINO a 4d input feature and
LeJEPA 2d against MAE's 1d — i.e. **different probe capacity per method**, which
confounds exactly the comparison this benchmark exists to make. So the headline
fixes both the representation and the head. The non-affine BN matters because
each SSL objective leaves an arbitrary per-feature scale in the embedding;
without it, probe accuracy partly measures that scale rather than linear
separability. `affine=False` keeps it pure whitening, so the only trained
parameters are the linear layer's and capacity is constant.

BarlowTwins/SimCLR/NNCLR do use BatchNorm inside their *projectors*, but those
are discarded before probing — a different thing from MAE's BN on the frozen
embedding.

**Ownership:** each model file supplies its own `build_probe(embed_dim,
num_classes, protocol)`, so method-specific knowledge lives next to that
method's `forward()`. Every one of them returns the common head under the
default `protocol="common"`. `conf/config.yaml: probe_protocol` selects it.

The head is fixed, but its OPTIMIZER hyperparameters are swept — see #14.

Native probes (`protocol="native"`) are available as an optional **secondary**
result. SimCLR/NNCLR/BarlowTwins natives are drop-in (bare Linear, same dim).
DINO and LeJEPA natives raise `NotImplementedError` with instructions, because
they additionally require `forward()` to emit a multi-layer CLS concatenation —
deliberately raising rather than silently building a mis-shaped head.

**Legacy path:** `protocol="legacy_linear"` reproduces the flat `nn.Linear` head
used before 2026-09. `recover_online_probe.py` is pinned to it. This matters:
that script reloads probe weights under `strict_loading = False`, which does not
raise on a shape mismatch — it silently keeps the random init and reports
garbage accuracy.

## 8. Storage: one scratch root, no Google Drive

Everything this benchmark writes lives under a single root:

```
~/scratch/stable-datasets-iclr/
├── downloads/        raw dataset archives
├── processed/        Lance/arrow shards (the dataset cache)
├── checkpoints/      {model}_{backbone}_{dataset}[_seed{N}]/
└── benchmark-runs/   hydra + submitit run dirs, local W&B files
```

Set from `${oc.env:HOME}` rather than a hardcoded username, so ian and leyang get
their own root with no config edit. `$STABLE_DATASETS_ROOT` overrides it.
`~/scratch` is a symlink to `/oscar/scratch/$USER`; both spellings are one dir.

**Google Drive is NOT used.** The original estimate of "a few TB" of checkpoints
was off by ~10x. A real ViT-S/16 checkpoint with optimizer state measures
**387 MB**, and `save_top_k: 1` + `save_last: true` keeps **2 files per cell**,
not a history:

| plan | ViT cells | RN-50 cells | total |
| --- | ---: | ---: | ---: |
| 1 seed | 140 | 120 | **211 GB** |
| uneven seed plan (2.3 avg) | 322 | 276 | **486 GB** |
| 3 seeds | 420 | 360 | 634 GB |

That fits on scratch, so the whole Drive apparatus — OAuth, per-collaborator
tokens, upload bandwidth, rate limits — was removed from the critical path.

`GoogleDriveModelCheckpoint` (`benchmarks/gdrive_checkpoint.py`) remains in the
tree, wired and tested, with `checkpoint.gdrive.enabled: false`. It is reserved
for post-deadline archival. If it is ever turned on, note its two invariants:
`last.ckpt` is never deleted locally while the run is live (SLURM preemption
resumes from it), and a local file is deleted only after a byte-verified upload.

Drive was rejected on two grounds beyond being unnecessary. Brown's Workspace
gives 100 GB, not the 5 TB assumed, so a Brown Shared Drive could not hold the
grid. And in a personal My Drive, **the uploader owns the bytes and they count
against the uploader's quota** — so pooling collaborators' writes into one shared
folder spends each collaborator's own storage anyway, which is why the
per-person-folder arrangement (and the `$SDS_WHOAMI` config keyed to it) was the
right shape to begin with.

**Alternative if scratch gets tight:** `/oscar/data/apma2822_24fall` is a group
allocation at 250 GB soft / 500 GB hard, currently empty. CCV grants increases on
request. Unlike scratch it is not subject to a purge policy — worth moving to if
the checkpoints need to outlive the sweep.

## 9. SLURM: requeue budget and GPU feature pinning

**`max_num_timeout: 3` -> `8`.** The hard wall on a cell is
`timeout_min * (1 + max_num_timeout)`. At 3 that was 96 h, which silently killed
the two longest cells mid-run: `imagenet100/dino` needs ~116 h and
`tissuemnist/dino` ~103 h at 400 epochs. 8 gives 216 h and leaves room for
preemption-driven requeues, which draw from the same budget.

**GPU feature pinning.** The `3090-gcondo` *partition* also contains RTX A5000
and A5500 nodes, which expose ~22.0 GiB against the 3090's 24 GiB. DINO and
LeJEPA sit right at that boundary. `conf/slurm.yaml` already pinned
`constraint: geforce3090|l40s`, but the `interact3090` shell alias requested the
partition with **no feature filter**, so interactive sessions landed on a 22 GiB
card at random and OOMed. The alias now passes `-f geforce3090`.

Note `conf/local_parallel.yaml` (the `launch.sh` default) uses the **joblib**
launcher, which runs on whatever node you are already on — no SLURM constraint
applies there at all. Use the `slurm` config for sweeps.

**Not the cause of OOM:** going from 1 to 9 online linear probes adds only
25–111 MB depending on backbone and class count, against the ~2,048 MB gap
between a 22 GiB and 24 GiB card. Probe count is not a memory concern.

## 10. Datasets are prefetched, not downloaded per-run

`benchmarks/prewarm.py` materializes every dataset cache up front so concurrent
SLURM workers hit a warm cache instead of racing on the download path (several
source URLs are unreliable).

It previously defaulted to a repo-relative `./.anonymous-datasets-cache`, which
populated a cache under HOME that **no training run ever read** — `run.py` passes
`cfg.data_dir` (`$STABLE_DATASETS_ROOT`, default `~/scratch/.stable-datasets`).
Now both resolve to the same scratch root. `~/scratch` is a symlink to
`/oscar/scratch/$USER`, so both spellings are one directory.

## 11. Compute budget (1 seed, ViT-S, the 20 datasets)

**3,060 GPU-hours.** By method: dino 699, lejepa 601, mae 520, barlow_twins 371,
simclr 337, nnclr 336, supervised 196. ResNet-50's 6-method grid is 2,752 GPU-h
(cheaper than ViT's 7-method grid despite being slower per epoch, because MAE
and its 4x multiplier drop out).

The binding constraint is **not** total throughput — it is the single longest
unsplittable job. `imagenet100/dino` is ~4.8 days on one GPU, so the makespan is
4.8 days at 40 GPUs, at 64, or at 200. Adding GPUs does not help; only cutting
epochs on that cell does.

## 12. Seeds — **OPEN**

3 seeds everywhere does not fit a 6-day deadline at 40 GPUs: 9,179 GPU-h (159%
of budget), 9.6-day makespan. Even 2 seeds everywhere misses at 6.4 days.

Proposed: uneven seeds, cheapest-first, capped at a 5-day makespan —
**4,735 GPU-h, 4.96 days, 2.3 seeds/dataset average**:

- 3 seeds (13): bloodmnist, cifar10, cifar100, cub200, dermamnist, dtd,
  fgvcaircraft, flowers102, galaxy10, imagenette, organamnist, pneumoniamnist,
  stl10
- 1 seed (7): country211, food101, imagenet100, octmnist, pathmnist, svhn,
  tissuemnist

Not yet accepted. Note the estimate assumes zero queue wait and no preemption
loss beyond what auto-resume recovers.

## 13. Bugs found in the hparam audit (2026-09-12)

### DINO's EMA teacher was never updated — all prior DINO results are invalid

`TeacherStudentWrapper` moves its teacher parameters **only** when something
calls `update_teacher()`; it does not self-update on forward. That call comes
from `spt.TeacherStudentCallback`, which was **never registered** in `run.py`.

So for every DINO run to date, the teacher stayed frozen at its warm-init copy of
the student's *random initial* weights, and DINO was distilling a fixed random
target rather than an EMA of itself. It trains, the loss falls, and the probe
numbers look plausible — which is exactly why it went unnoticed.

Measured through a real Lightning loop with `accumulate_grad_batches=8`:

| | teacher drift | final EMA coeff |
| --- | --- | --- |
| without the callback (old) | **0.0** | 0.996 (static) |
| with the callback (fixed) | 2.09e-3 | 0.999 (annealing to 1.0) |

Fixed in `run.py` by registering the callback whenever the module contains
anything with an `update_teacher` method — duck-typed, so a future EMA method
picks it up automatically. The callback's defaults are correct under gradient
accumulation: it fires on `on_train_batch_end` guarded by `trainer.global_step`,
which advances once per *optimizer* step, so DINO/LeJEPA's accum=8 gets one EMA
update per optimizer step, not eight.

**Any DINO number produced before this fix should be discarded.**

### DINO's teacher-temperature warmup was a no-op

```yaml
temperature_teacher: 0.04
warmup_temperature_teacher: 0.04     # start == end -> schedule does nothing
warmup_epochs_temperature_teacher: 301
```

Start equalled end, so the warmup was dead, and its 301-epoch duration exceeded
the total schedule of several datasets (fgvcaircraft runs 200). Set to DINO's
reference 0.04 -> 0.07 over 30 epochs; 30 is shorter than our shortest schedule,
so the warmup always completes. The EMA is now explicitly annealed
`momentum_teacher: 0.996` -> `final_momentum_teacher: 1.0` rather than relying on
a library default.

### The supervised baseline's batch size was neither constant nor matched

`supervised.yaml` carried `batch_size: 128` on 12 of the 20 datasets and 256 on
the other 8, against every SSL method's uniform effective 256 — with a fixed LR
of 5e-4 either way, so LR-per-sample silently varied *by dataset within the same
method*. In a benchmark whose claim is that only the method varies, the baseline
was the one uncontrolled term. Pinned to 256 everywhere.

After these fixes all 7 methods run at effective batch 256 on all 20 datasets
(DINO/LeJEPA as 32 x 8), with no remaining anomalies.

### Checked and found already safe

- **BatchNorm1d probe on a 1-sample batch.** `organamnist` has 34,561 training
  images, which is 1 mod 256 *and* 1 mod 32, so its last batch holds exactly one
  sample — and `BatchNorm1d` raises in training mode on that. It never fires:
  `train_drop_last` defaults to `True` (`dataset.py`), and validation runs under
  `eval()` where BN uses running statistics.
- **Multicrop OOM.** No dataset resolves DINO or LeJEPA to `accum=1`; both are
  32 x 8 across all 20.

## 14. The probe's own LR and weight decay are swept, not fixed

Probe accuracy depends on the probe's learning rate and weight decay, and the
best setting is not the same for every method: SSL objectives leave embeddings
at very different scales. Fixing one probe LR for all methods measures "how well
does this embedding happen to suit our arbitrary probe LR" alongside linear
separability — the exact confound the benchmark exists to remove.

So `K = 3 x 3 = 9` heads train simultaneously on the same frozen embedding:

```yaml
probe_sweep:
  enabled: true
  lr_scales:     [0.1, 1.0, 10.0]      # multiply the probe optimizer's base LR
  weight_decays: [0.0, 1.0e-6, 1.0e-4]
  per_head_metrics: true
```

The head itself is unchanged and identical for every method (#7): non-affine
BatchNorm + Linear. Only the optimizer settings vary, so probe *capacity* stays
constant and only probe *fitting* is tuned.

**How it works.** `SweepLinearProbe` stacks K heads and returns `(N, K, C)`.
Because the embedding is detached, the heads are independent: their
cross-entropies are summed, and one optimizer step updates all K as if each had
been trained alone. Per-head LR/WD come from a gradient hook that scales each
head's gradient by `lr_scale` and adds `weight_decay * param` — one optimizer,
K effective settings. Verified: a head at `lr_scale=10` moves 108x more per step
than one at `lr_scale=0.1` (expected ~100x, the remainder being weight decay).

**Metric semantics — read this before interpreting results.** `top1` and `top5`
keep their names and now report the **best head**, so every existing consumer of
`eval/linear_probe_top1_epoch` keeps working without knowing a sweep happened.
`top1_mean` is the mean across heads, and each head is also logged as
`eval/linear_probe_lr<x>_wd<y>_top1`. Keep the per-head metrics on: they are how
you see whether the best head sits at the EDGE of the grid, which is the signal
that the grid needs widening.

Best-of-K selected on validation is a mild optimistic bias. It is the standard
linear-eval protocol (DINOv2 and others sweep probe LR and report the best), and
it is applied identically to every method and backbone, so between-method
comparisons stay fair.

Cost is negligible: 9 heads add ~25-110 MB depending on backbone and class
count, against the ~2 GB gap between a 22 GiB and a 24 GiB card.

**Why not `spt.backbone.AutoLinearClassifier`.** It does almost this, but it has
no usages or tests anywhere in spt, and three things disqualified it as a
drop-in: it hardcodes `BatchNorm1d(d)` with `affine=True, eps=1e-5` rather than
our `affine=False, eps=1e-6`; its default grid also sweeps normalization,
dropout and label smoothing, which would reopen #7 by making the headline a best
over *architectures* rather than one protocol; and its `forward(x, y, pl_module)`
returns a summed loss rather than logits and does not detach its input, so it
cannot go through `spt.callbacks.OnlineProbe` (which does detach for us) without
a rewrite. `benchmarks/models/probe_sweep.py` is ~175 lines and keeps the
arithmetic behind every probe number in this repo.
