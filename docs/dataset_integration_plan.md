# Dataset Integration: Sign Language MNIST and Chars74K EnglishHnd

## Overview

This contribution will integrate two image classification
datasets into stable-datasets.

## Dataset 1: Sign Language MNIST

- 24 American Sign Language letter classes
- 28x28 grayscale images
- Official train/test splits
- CSV data format

Source:
https://www.kaggle.com/datasets/datamunge/sign-language-mnist

## Dataset 2: Chars74K EnglishHnd

- 62 handwritten alphanumeric classes
- 55 samples per class
- PNG image files
- Reproducible stratified train/test split

Source:
https://teodecampos.github.io/chars74k/

## Planned Work

- Implement both dataset builders
- Add dataset metadata and source information
- Implement data loading and preprocessing
- Add unit tests and smoke tests
- Run training and evaluation experiments
- Document reproducible results

Status: Work in progress.
