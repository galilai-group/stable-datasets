# Scene8 and ISIC2016 Dataset Integration

## Objective

Add two image datasets to stable-datasets, test their loading behavior, and provide a working training/evaluation example using an established self-supervised model.

| Deliverable | Implementation and evidence |
| --- | --- |
| Two native datasets | `stable_datasets/images/scene8.py` and `isic2016.py`, with public imports and dataset documentation |
| Code correctness | 44 passing offline tests covering the builders, data partitions, and validation pipeline |
| Training/evaluation setup | `benchmarks/transfer/primary_scene8_isic2016.py` fits and evaluates a linear classifier on frozen DINO features; a separate SimCLR smoke check verifies encoder updates |

## Dataset implementation

| Dataset | Original data | Builder behavior |
| --- | --- | --- |
| Scene8 | 2,688 outdoor images, 8 classes | Exposes the complete original collection; experiment splits are defined separately |
| ISIC2016 Task 3 | 900 training and 379 test images, 2 classes | Preserves the official source split and joins CSV labels by image ID |

Both builders return `image`, `label`, and `image_id`, preserve original JPEG bytes, support local source files, and use the library's normal cache interface.

```python
from stable_datasets.images import Scene8, ISIC2016

scene = Scene8(split="train")
isic = ISIC2016()
sample = isic["train"][0]
print(sample["image_id"], sample["label"], sample["image"].size)
```

## Correctness checks

All 44 tests in the four focused test files pass. They check label matching, invalid inputs, stable IDs, original image bytes, reopened caches, partition separation, and classifier evaluation. This count covers the focused suite, not the full repository suite. A separate source audit verified all 3,967 images.

Exact duplicate images stay together in Scene8's experimental splits. Three ISIC training images duplicate official test images: the builder retains them, while the primary validation excludes them from its training pool. The official test set stays at 379 images. The audit uses test pixels to identify these duplicates, without using test diagnoses or model scores. It does not establish patient independence.

## Main validation

The pipeline is: load images, extract features with the original ImageNet-pretrained DINO ViT-S/16, train a linear classifier, and evaluate held-out images. DINO remains frozen; only the classifier is fitted. This is transfer validation, not DINO pretraining on the new datasets.

The classifier selects its regularization on validation data, then refits on the allowed fitting/validation union. Test labels are used only for evaluation. The table uses all full-label linear runs at the three previously fixed partition seeds, 42, 43, and 44. The same method is used for both datasets.

| Dataset | Primary metric | Result |
| --- | --- | ---: |
| Scene8 | Balanced accuracy | 0.9650 +/- 0.0072 |
| ISIC2016, duplicate-filtered training | Average precision (AP) | 0.6481 |

Scene8 balanced accuracy averages recall across its eight classes; the displayed variation is sample SD across overlapping partitions. ISIC AP measures ranking of malignant images, and its constant-score baseline is 0.1979. Its full-label linear refits use the same 897-image pool and select the same parameter, so their predictions coincide across inner-partition seeds.

At the fixed 0.5 threshold, the ISIC classifier identifies 32 of 75 malignant test images. Ranking performance and overall accuracy should therefore not be interpreted as high malignant recall. Exact-pixel checks do not rule out near-duplicates or overlap with encoder pretraining.

## Reproduction

The short workflow is documented in `benchmarks/transfer/SCENE8_ISIC2016_README.md`. After the data audit and feature extraction, `python -m benchmarks.transfer.primary_scene8_isic2016` runs only this main validation. An additional short SimCLR optimization check is recorded separately as a training-path check, not a convergence result.

[Supplementary verification and full experimental records](SUPPLEMENT.md)

The supplement retains the earlier extended comparisons for transparency; they are outside the main presentation.

```{toctree}
:hidden:

SUPPLEMENT
```
