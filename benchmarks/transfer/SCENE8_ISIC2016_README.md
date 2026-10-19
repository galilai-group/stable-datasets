# Scene8 and ISIC2016: Main Validation

The main contribution is two native dataset builders, their tests, and one
working validation method: frozen DINO ViT-S/16 features plus a trained linear
classifier. Start with the dataset pages and the short integration report.

## Environment

Use a dedicated Python environment. The executed setup used Python 3.12.14:

```console
python -m pip install torch==2.9.1+cpu torchvision==0.24.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e . pytest
```

Executed package versions are retained in the report's environment.txt. On
Windows, use .venv\Scripts\python.exe or activate that environment first.

## Reproduction

Place the original Scene8 ZIP and four original ISIC2016 ZIP/CSV files in
.cache/project1/raw, using the names documented in the dataset pages. Then:

```console
python -m benchmarks.audit_scene8_isic2016 --raw-dir .cache/project1/raw --cache-dir .cache/project1/processed --output-dir .cache/project1/verified-v2
python -m benchmarks.transfer.scene8_isic2016 extract --threads 4 --batch-size 4
python -m benchmarks.transfer.primary_scene8_isic2016
```

The audit checks source files and prepares explicit partitions. The extractor
loads the pinned original DINO checkpoint, freezes it, and saves one feature
vector per image. The final command runs only the full-label linear classifier
on both datasets at the three previously declared partition seeds. It writes
predictions, selected parameters, metrics, and a short report to
.cache/project1/primary-validation. Existing verified feature caches are reused.

Select classifier parameters on fitting/validation data, then refit on their
union and evaluate test. Scene8 has project-defined partitions. ISIC keeps all
379 test images; three duplicate training images are excluded from its primary
validation pool. The builder itself preserves all original records.

## Tests

```console
python -m pytest stable_datasets/tests/images/test_scene8.py stable_datasets/tests/images/test_isic2016.py stable_datasets/tests/test_scene8_isic2016_protocols.py stable_datasets/tests/test_manifest_probes.py -q
```

Tests use tiny offline fixtures and synthetic embeddings. They check labels,
images, caching, partitions, and the classifier workflow without downloading
images or model weights.

## Supplementary material

SCENE8_ISIC2016_SUPPLEMENT.md documents the earlier extended checks and the
separate SimCLR optimization smoke run. The report's SUPPLEMENT.md retains all
original comparisons. These are optional supporting evidence, outside the main
presentation. No earlier results have been removed or changed.
