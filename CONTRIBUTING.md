# Contributing to stable-datasets

## Adding a dataset

Before you start, search `stable_datasets/` and the open pull requests for the dataset, so you don't duplicate someone else's work.

A dataset pull request includes:

1. **The builder**: `stable_datasets/<modality>/<dataset>.py`, a subclass of `BaseDatasetBuilder` (see "Onboarding new datasets" in the README), exported from that folder's `__init__.py`. Public classes, methods and functions need docstrings. Name helper modules with a leading underscore, e.g. `_my_utils.py`.
2. **Tests**: `stable_datasets/tests/<modality>/test_<dataset>.py`, checking split sizes, example keys and types, and label ranges. See "Tests" below for which markers to use.
3. **Docs**: a page at `docs/source/datasets/<dataset>.rst`, listed in `docs/source/datasets/index.rst`, with a teaser figure in `docs/source/datasets/teasers/`. `python generate_teaser.py --name <DatasetClass> --output <dataset>_teaser.png` makes one for image datasets; for other data, make the figure however suits it.
4. **Results**: in the pull request description, a metric (e.g. train/val accuracy) from training or evaluating a model on the dataset, with supervised learning or any self-supervised method.

## Tests

Run the tests that CI runs, or a single dataset's tests:

```bash
pytest -m "not large"
pytest stable_datasets/tests/images/test_<dataset>.py
```

Mark tests by what they need:

- `@pytest.mark.large`: the test downloads more than about 10 GB. These never run in CI, so the reviewer runs them by hand before merging.
- `@pytest.mark.skip(reason="...")`: the data needs a login or a manual download (e.g. ImageNet or gated ISIC data), so the test can't run automatically anywhere.

Unknown markers are an error, so a typo like `@pytest.mark.larg` fails right away instead of running a huge download. Declare any new marker in `pyproject.toml`.

Let builders download to the default cache, which CI clears after each test, or pass pytest's `tmp_path` as the download directory.

## What CI checks

On every pull request:

- **pre-commit**: formatting and linting with ruff (including docstrings on dataset builders) and a 1 MB limit on added files.
- **Testing**: the tests for the datasets you changed. If you change shared code (anything else under `stable_datasets/`, or `pyproject.toml`), the whole suite runs instead. Tests marked `large` are skipped. In the whole-suite run, a test whose dataset host can't be reached is reported as skipped rather than failed, since it says nothing about your change; for the datasets you changed, it fails, so re-run the check once the host is back.
- **Docs**: the docs are built, and the check fails on any Sphinx warning.
- **Dataset checklist**: each new builder has a test file and a docs page, and the page is listed in the index.

A maintainer has to approve CI runs for first-time contributors, so checks may not start right away.

Every night, the whole suite runs on `main`. If any dataset breaks, usually because its host went down or moved, the run opens an issue titled "Nightly dataset tests are failing" and closes it once the run passes again.

## Create a pull request

- Start the title with what it adds, e.g. `[Dataset] Add GTSRB`, `[Feature] ...`, `[Bugfix] ...` or `[Docs] ...`.
- While it isn't ready for review, add `[WIP]` to the title or open it as a draft.
- Fill in the pull request template, including your results.
- Run `pre-commit run --all-files` before pushing, or `pre-commit install` to run it on every commit.
