# Scene8 and ISIC2016: Pre-Model Experiment Protocol

Status: design reviewed before any model performance results. This is a project
protocol, not a claim of instructor approval or an external preregistration.
Its implementation and executed evidence must be distinguished from this design.

## 1. Questions and scope

1. Can two native dataset builders preserve the original sources and support a
   reproducible training/evaluation workflow with an established SSL encoder?
2. How do linear and nearest-neighbor classifiers on the same frozen features
   behave under decreasing target-task label budgets?
3. How sensitive are ISIC results to its three exact train/test duplicates?

Use the original DINO ImageNet-1K ViT-S/16 as the fixed encoder. These are frozen
transfer experiments, not target-dataset DINO pretraining or a reproduction of
the DINO paper's ImageNet linear-evaluation numbers. No test result may determine
which dataset, protocol, encoder, fraction, seed, or metric is reported.

## 2. Decisions made during the protocol review

| Review finding | Decision |
| --- | --- |
| Removing three ISIC images changed many validation assignments under v1 | V2 shares a validation set and all retained assignments across the pair |
| Validation labels could be omitted from a low-label budget | Subsample both fitting and validation roles; count their union |
| Tuning a probability threshold and then refitting can shift calibration | Use a predeclared threshold of 0.5; do not optimize it |
| AUROC and AP had no single selection priority | ISIC selects by AP; Scene8 selects by balanced accuracy |
| Existing 384-dimensional DINO features differ from published multi-block linear evaluation | Use the same final-block CLS features for both probes and describe the adaptation |
| Repeated seeds do not create independent test sets or independent pretrained models | Report what varies and avoid treating seed SD as a confidence interval |

The v1 official-versus-filtered validation symmetric differences were 274, 280,
and 296 IDs for seeds 42, 43, and 44 respectively. No scores were examined to
make this correction. Keep the v1 data audit as historical evidence and generate
new manifests with explicit v2 names.

## 3. Source-preserving builders and experimental roles

Never change labels, deduplicate, crop, or resize inside the builders. All
experimental assignments live in saved manifests. `fit`, `validation`, and
`test` are disjoint. `trainval` is the union of fitting and validation IDs and
is available for the final classifier refit after hyperparameter selection.

### Scene8

Use `scene8-pixel-groups-v1`, for seeds 42, 43, and 44. Five-fold grouped
stratification keeps exact RGB duplicates together. Fold 0 is test, fold 1
validation, and the other folds fit the classifier. This is approximately
60/20/20, not an official source split. Preserve the conflicting-label duplicate
pair and disclose it; do not remove it after observing prediction errors.

Seed 42 has 1,613 fitting, 537 validation, and 538 test images. The refit pool
contains 2,150 images. The different seeds vary the experimental partitions;
their test sets overlap and must not be treated as independent replications.

### ISIC2016

Use `isic2016-train-filtered-pixel-groups-v2` for the primary analysis. Remove
every official training image with exact decoded pixels found in official test
before creating the shared fitting/validation assignment. The excluded IDs are
`ISIC_0000280`, `ISIC_0001184`, and `ISIC_0001385`. The test set remains all 379
official test images. The eligible pool has 897 images; seed 42 has 717 fitting
and 180 validation images.

For the full-budget sensitivity analysis, use
`isic2016-official-pixel-groups-v2`. Reuse those same validation IDs and put the
three excluded records back into fitting. The official refit pool has all 900
images. It preserves the official train/test membership but the inner validation
partition is still ours. Both variants use test pixels to coordinate the audit;
this transductive integrity check must be disclosed. Test diagnoses and scores
do not influence assignments. Patient independence is not established.

For the paired comparison, reuse the hyperparameter selected on the filtered
variant at 100% budget, then refit on 897 versus 900 images and predict the same
test IDs. Do not retune the official companion. The resulting change describes
this specific duplicate-inclusion sensitivity, not a universal or causal estimate
of leakage. The official companion is not the clean primary result.

Save complete manifests, their SHA-256 hashes, splitter version, and source
fingerprints. A random seed alone is insufficient for cross-version reproduction.

## 4. Label budgets and nested subsets

Use nominal fractions 0.10, 0.25, and 1.00 of each primary fitting and validation
role. Keep the test set fixed across fractions within each partition seed.
Each fitting subset must be nested in the next fitting subset; apply the same
rule separately to validation. Both probes use exactly the same subsets.

The implementation contract for reproducible grouped subsampling is:

1. Within each role, group all records by exact RGB pixel hash. Never split a
   duplicate group, including a group with conflicting labels.
2. Assign each group a sampling stratum equal to its smallest source class ID.
   This is only a sampling key: retain every original per-image target. For the
   single mixed-label Scene8 group, report this explicit convention.
3. Within each stratum, order groups by the hexadecimal SHA-256 of a canonical
   JSON list `["label-budget-v1", subset_seed, role, pixel_sha256]`, serialized
   with `separators=(",", ":")` and UTF-8. Break a hash tie by pixel hash.
   Use subset seeds 1042, 1043, and 1044 paired with partition seeds 42, 43, and 44.
4. Select the first `ceil(fraction * number_of_groups_in_stratum)` groups.
   Include all their images. At fraction 1.00 this must recover the entire role.
5. Require all task classes in both selected fitting and validation roles.
   Fail visibly on an invalid budget rather than retrying seeds until it works.

Whole groups and per-stratum rounding make the actual image fraction approximate.
Persist all selected IDs, group counts, class counts, the nominal fraction, and
the actual number and fraction of distinct labeled images. Use actual labeled
image counts on plots. The label budget is
`len(selected_fit union selected_validation)`, not only `len(selected_fit)`.

Select hyperparameters using only the selected fitting and validation labels.
Refit the chosen classifier on their union, not on the complete parent trainval
pool. A 10% run cannot reuse tuning decisions from a higher-budget run. Keep
unused target labels inaccessible to classifier fitting and model selection.
The data-splitting coordinator uses labels for stratification, as in a supervised
benchmark; this is not a simulation of prospective label acquisition.

With approximately 18 validation images at the smallest ISIC budget, AP-based
selection will be noisy. Keep the fixed small search grid, disclose class counts,
and report all seeds without searching for more favorable partitions.

## 5. Encoder, preprocessing, and CPU feasibility

Use the official DINO ViT-S/16 pretrained backbone at:

`https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth`

Record its measured SHA-256 and byte count; a measured hash is not a publisher
signature. Load compatible backbone weights without ignored trunk keys, freeze
every parameter, and use evaluation/inference mode. Pin the implementation and
dependency versions. Verify feature parity with the pinned original DINO
implementation on a fixed input before accepting an alternative implementation.

Use RGB, bicubic shorter-edge resize to 256, center crop to 224 by 224,
conversion to float in [0,1], and ImageNet mean/std normalization
`(0.485,0.456,0.406)/(0.229,0.224,0.225)`. Record interpolation, antialias behavior,
Pillow/torchvision versions, and preprocessing code hash. Inspect a training-only
preprocessing montage, especially for lesion cropping, before any model scores.
No augmentation or test-time ensemble is used for this fixed-feature protocol.

Extract the final normalized CLS token, 384 dimensions, for both probes. The
original DINO linear-evaluation script concatenates four blocks for ViT-Small
and trains with augmentation; our common-feature transfer comparison deliberately
has a different protocol. The final-block choice is consistent with the original
DINO kNN feature path. Do not label this a reproduction of published linear scores.

Start with CPU FP32, zero loader workers, and small batches. Time a 64-128-image
training-only pilot, including image decoding, and report its throughput and
memory alongside a qualified full-run estimate. Choose batch size/thread count
from resource measurements, never from test accuracy. If infeasible, revise and
version the protocol before seeing performance results rather than silently
changing resolution or the encoder.

## 6. Feature and result identity

Extract every source image exactly once with stable IDs, `shuffle=False`, and
`drop_last=False`, including the final partial batch. Read source datasets by
bounded indexing to avoid bulk decoding of high-resolution Arrow batches.
Assert unique IDs, exact expected coverage, valid labels, finite 384-dimensional
features, and no zero-norm vectors. Store original unnormalized embeddings and
apply per-row L2 normalization consistently to both probes. No population
standardization, PCA, or adaptation on validation/test features is allowed.

A complete-source frozen-feature cache may be reused across manifests: per-image
inference has no fitted target-data statistics. Its identity must bind source
release/fingerprint, builder version, complete ordered IDs, weight hash,
architecture/feature layer, preprocessing, implementation hashes, and relevant
versions. Save ID/label/embedding alignment and verify the cache payload hash.
Do not rely on a dataset/backbone directory name alone.

The experiment-result identity additionally binds the parent and subset manifests,
configuration, selected parameters, classifier implementation, seeds, and feature
cache identity. Changing a split must invalidate a result without unnecessarily
recomputing identical per-image frozen features. Partial writes must not be
accepted as complete caches or completed results.

## 7. Classifiers, model selection, and metrics

| Method | Fixed setup | Validation choices |
| --- | --- | --- |
| Prior baseline | Constant class probabilities from the current fitting labels; refit on the allowed union | None |
| Linear probe | Logistic regression on L2-normalized embeddings, intercept, L2 regularization, no class weighting, LBFGS, tolerance 1e-6, maximum 5,000 iterations | C in [0.1, 1.0, 10.0] |
| Weighted cosine kNN | L2-normalized embeddings, cosine neighbor ranking, weights proportional to exp(similarity / 0.07), normalized class votes | k in [5, 10, 20], restricted to k <= fitting count |

Select separately for each dataset, partition seed, and budget. Use balanced
accuracy (macro recall) for Scene8 and average precision for ISIC. Choose the
first candidate in the listed ascending grid when scores are tied within 1e-12.
Record every validation candidate score before test evaluation. Reject convergence
warnings or nonfinite outputs; do not silently report a failed fit as a result.
Neighbor ties use sorted image ID as the deterministic secondary key.

After selection, refit on the permitted fit/validation union and then evaluate
test. For ISIC use malignant probability/vote as the positive score. Use the
fixed rule `score >= 0.5` for binary class predictions. Multiclass prediction uses
argmax, with the smallest class ID on a tie. No optimized clinical operating
threshold or calibration claim is made.

Report:

- Scene8: **balanced accuracy** as primary, top-1 accuracy, macro F1, per-class
  recall, and the confusion matrix. Use explicit names; avoid ambiguous
  expressions such as "macro accuracy."
- ISIC: **AP** as primary, AUROC, balanced accuracy, macro F1, sensitivity,
  specificity, accuracy, and the confusion matrix at threshold 0.5.

Use scikit-learn `average_precision_score`, its non-trapezoidal AP definition,
and record the version. Do not relabel trapezoidal PR-AUC as AP. The original
ISIC task ranks by AP, but this work does not establish exact numerical
equivalence to the historical challenge scorer. These experiments do not
establish clinical validity.

Use `zero_division=0` for undefined per-class precision/F1 and report class
support. Constant-score ISIC predictions have AUROC 0.5 and AP equal to the
evaluated positive prevalence; 304/379 majority accuracy alone is not evidence
of useful melanoma discrimination. Keep probabilities and original labels for
every test ID so all reported metrics can be independently recomputed.

## 8. Experiment matrix and uncertainty

Run two primary datasets, three partition/subset-seed pairs, three budgets,
and both learned probes: 36 learned-model conditions. Add the prior baseline to
each of the 18 dataset/seed/budget combinations. The full-budget ISIC official
companion contributes six more learned conditions, with frozen primary-selected
parameters. Total: 42 learned conditions plus 18 baselines, using one encoder.

Publish all per-seed values and their mean/sample SD. For Scene8, seeds vary
overlapping partitions and label subsets. For ISIC, seeds vary the inner
partition and subsets while the same official test set is reused. A deterministic
classifier rerun on unchanged inputs is not another independent seed.

For the seed-42, full-budget results, optionally add a declared conditional 95%
percentile interval from 2,000 bootstrap replicates with seed 20261018. Resample
exact-pixel groups within strata defined by the sorted multiset of group labels,
retaining all images in each sampled group. For paired model differences use
the same resampled groups for both models. This reflects finite test-sample
uncertainty conditional on the fitted models, group definition, and observed
strata; it excludes patient, training, partition, and pretraining uncertainty.
No hypothesis test or causal claim is required.

## 9. Implementation gates and evidence

Before formal evaluation, verify:

- Source, manifest, selected-ID coverage, role isolation, duplicate grouping,
  nested subsets, label counts including validation, and no full-budget label
  access from a low-budget run.
- Exact equality of ISIC paired validation sets and retained assignments;
  flipping test diagnoses cannot change role assignment.
- Extraction of a partial final batch, ID/label alignment, deterministic
  preprocessing, checkpoint compatibility, and feature-cache invalidation.
- Validation-only selection, subset-only refitting, fixed binary threshold,
  class order, known hand-checkable AP/AUROC/confusion examples, and prediction
  files that reproduce reported metrics.
- A bounded known-method training smoke test after the stable-pretraining
  dependency revision is pinned and verified. Distinguish a finite loss/gradient
  test from evidence of meaningful representation learning.

The generic existing benchmark test fallback is unsuitable for these protocols.
Provide a documented explicit-manifest entry point; do not merely register the
datasets and inherit implicit train/test behavior. Keep optional heavy training
dependencies out of data-audit imports and offline dataset tests.

Save configurations, source and weight hashes, all manifests, selected
hyperparameters, validation scores, all test predictions, metrics, timing,
versions, and executed source hashes including uncommitted changes. Distinguish
data-only audit evidence from model results. Repair any genuine implementation
bug found later, but disclose the revision and rerun the full affected matrix.

## 10. Presentation and remaining limitations

The central contribution is the two source-faithful dataset integrations and
their auditable evaluation. Present the original-source duplicate discovery,
the matched sensitivity protocol, label accounting, and actual experiments as
evidence. Report weak results and representative errors with the same care as
strong results. Do not claim absence of all leakage, patient independence,
pretraining disjointness, or guaranteed grades.

The instructor's interpretation of frozen transfer versus target-dataset SSL
pretraining remains to be confirmed. The repository's known-method smoke test
and transfer experiments address different parts of the evidence and should
be described separately. Final submission still needs a reviewable PR and the
course deliverables; neither has been sent by this workflow.

## Primary references

- [DINO original linear evaluation](https://github.com/facebookresearch/dino/blob/main/eval_linear.py)
- [DINO original weighted kNN evaluation](https://github.com/facebookresearch/dino/blob/main/eval_knn.py)
- [ISIC2016 Task 3 and ranking criterion](https://challenge.isic-archive.com/landing/2016/39/)
- [scikit-learn AP definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
