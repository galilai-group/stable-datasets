<div align="center">

# stable-datasets

_All dataset utilities (downloading/loading/batching/processing) in Numpy_

</div>


This is an under-development research project, not an official product, expect bugs and sharp edges; please help by trying it out, reporting bugs.
[**Reference docs**](https://rbalestr-lab.github.io/stable-datasets/)

## What is and why doing  ?

* First, stable-datasets offers out-of-the-box dataset download and loading only based on Numpy and core Python libraries.
* Second, stable-datasets offers utilities such as (mini-)batching a.k.a looping through a dataset one chunk at a time, or preprocessing techniques that are highly suited for machine learning and deep learning pipelines.
* Third, stable-datasets offers many options to transparently deal with very large datasets. For example, automatic mini-batching with a priori caching of the next batch, online preprocessing, and the likes.
* Fourth, stable-datasets does not only focus on computer vision datasets but also offers plenty in time-series datasets, with a constantly groing collection of implemented datasets.

## Minimal Example

```python
import stable_datasets as sds

mnist = sds.images.mnist.load()
train_images = mnist['train_set/images']
train_labels = mnist['train_set/labels']
```

## Installation

Installation is direct with pip as described in this [**guide**](https://rbalestr-lab.github.io/stable-datasets/).

# Datasets

## Classification

- CIFAR10
- CIFAR10-C
- CIFAR100
- CIFAR100-C
- TinyImagenet
- TinyImagenet-C
- MedMnistv2.1
