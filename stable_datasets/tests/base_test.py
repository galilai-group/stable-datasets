# def test_import():
#     import stable_datasets
#
# def test_CIFAR10():
#     import stable_datasets
#     a = stable_datasets.images.CIFAR10("../Downloads")
#     a.download()
#     a.load()
#
#
# def test_CIFAR100():
#     import stable_datasets
#     a = stable_datasets.images.CIFAR100("../Downloads")
#     a.download()
#     a.load()
#
#
# def test_Flowers102():
#     import stable_datasets
#     a = stable_datasets.images.Flowers102("../Downloads")
#     a.download()
#     a.load()
#
# def test_Food101():
#     import stable_datasets
#     a = stable_datasets.images.Food101("../Downloads")
#     a.download()
#     a.load()
#
# def test_FGVCAircraft():
#     import stable_datasets
#     a = stable_datasets.images.FGVCAircraft("../Downloads")
#     a.download()
#     a.load()
#
#
# def test_IBeans():
#     import stable_datasets
#     a = stable_datasets.images.Beans("../Downloads")
#     a.download()
#     a.load()
#
#
# def test_Country211():
#     import stable_datasets
#     a = stable_datasets.images.Country211("../Downloads")
#     a.download()
#     a.load()
#
#
# def test_MNIST():
#     import stable_datasets
#     a = stable_datasets.images.MNIST("../Downloads")
#     a.download()
#     a.load()
#
#
# def test_FashionMNIST():
#     import stable_datasets
#     a = stable_datasets.images.FashionMNIST("../Downloads")
#     a.download()
#     a.load()
#
#
# def test_CUB200():
#     import stable_datasets
#     a = stable_datasets.images.CUB200("../Downloads")
#     a.download()
#     a.load()
#
# def test_SVHN():
#     import stable_datasets
#     a = stable_datasets.images.SVHN("../Downloads")
#     a.download()
#     a.load()
#
# def test_RockPaperScissors():
#     import stable_datasets
#     a = stable_datasets.images.RockPaperScissors("../Downloads")
#     a.download()
#     a.load()
#
# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--name", type=str)
#     parser.add_argument("--path", type=str)
#     args = parser.parse_args()
#
#     import stable_datasets
#     a = stable_datasets.images.__dict__[args.name](args.path)
#     a.download()
#     a.load()
