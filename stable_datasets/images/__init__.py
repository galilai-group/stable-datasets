#!/usr/bin/env python

from .arabic_characters import ArabicCharacters
from .cars196 import Cars196
from .cifar10 import CIFAR10

# from .arabic_digits import ArabicDigits
# from .awa2 import AWA2
# from .beans import Beans
# from .celeb_a import CelebA
# from .cifar10 import CIFAR10
from .cifar10_c import CIFAR10C
from .cifar100 import CIFAR100

# from .cifar100 import CIFAR100
from .cifar100_c import CIFAR100C
from .clevrer import CLEVRER
from .dtd import DTD
from .dtd import DTD
from .fashion_mnist import FashionMNIST
from .k_mnist import KMNIST
from .med_mnist import MedMNIST

# from .country211 import Country211
# from .cub200 import CUB200
# from .dsprites import DSprites
# from .dtd import DTD
# from .e_mnist import EMNIST
# from .fashion_mnist import FashionMNIST
# from .fgvc_aircraft import FGVCAircraft
# from .flowers102 import Flowers102
# from .food101 import Food101
# from .hasy_v2 import HASYv2
# from .imagenette import Imagenette
# from .linnaeus5 import Linnaeus5
# from .mnist import MNIST
from .not_mnist import NotMNIST


# from .places365_small import Places365Small
# from .rock_paper_scissor import RockPaperScissor
# from .stl10 import STL10
# from .svhn import SVHN
# from .tiny_imagenet import TinyImagenet
# from .tiny_imagenet_c import TinyImagenetC

__all__ = [
    "ArabicCharacters",
    "CIFAR10",
    "CIFAR100",
    "CIFAR10C",
    "CIFAR100C",
    "CLEVRER",
    "FashionMNIST",
    "KMNIST",
    "MedMNIST",
    "Cars196",
    "DTD",
    "NotMNIST",
]
