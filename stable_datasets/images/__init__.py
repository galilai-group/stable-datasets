#!/usr/bin/env python

from .aiderv2 import AIDERv2
from .arabic_characters import ArabicCharacters
from .arabic_digits import ArabicDigits
from .awa2 import AWA2
from .beans import Beans
from .bee_dataset import BeeDataset
from .belgium_tsc import BelgiumTSC
from .breakhis import BreakHis
from .bsds500 import BSDS500
from .caltech101 import Caltech101
from .caltech101_silhouettes import Caltech101Silhouettes
from .caltech256 import Caltech256
from .cars3d import Cars3D
from .cars196 import Cars196
from .cct20 import CCT20
from .cifar10 import CIFAR10
from .cifar10_1 import CIFAR10_1
from .cifar10_2 import CIFAR10_2
from .cifar10_c import CIFAR10C
from .cifar100 import CIFAR100
from .cifar100_c import CIFAR100C
from .cinic10 import CINIC10
from .clevrer import CLEVRER
from .cmu_faces import CMUFaces
from .coil20 import COIL20
from .coil100 import COIL100
from .country211 import Country211
from .cub200 import CUB200
from .deep_weeds import DeepWeeds
from .defungi import DeFungi
from .devanagari import Devanagari
from .dsprites import DSprites
from .dsprites_color import DSpritesColor
from .dsprites_noise import DSpritesNoise
from .dsprites_scream import DSpritesScream
from .dtd import DTD
from .e_mnist import EMNIST

# from .celeb_a import CelebA
from .eth80 import ETH80
from .eurosat import EuroSAT
from .face_pointing import FacePointing
from .fashion_mnist import FashionMNIST
from .fgvc_aircraft import FGVCAircraft
from .flowers17 import Flowers17
from .flowers102 import Flowers102
from .fmd import FMD
from .food101 import Food101
from .freiburg_groceries import FreiburgGroceries
from .fruits360 import Fruits360
from .galaxy10 import Galaxy10Decal
from .grocery_store import GroceryStore
from .gtsrb import GTSRB
from .hasy_v2 import HASYv2
from .horse_or_human import HorseOrHuman
from .imagenet_1k import ImageNet1K
from .imagenet_10 import Imagenette
from .imagenet_100 import ImageNet100
from .imagenet_a import ImageNetA
from .imagenet_r import ImageNetR
from .imagenet_sketch import ImageNetSketch
from .imagenet_v2 import ImageNetV2
from .imagewoof import Imagewoof
from .intel_image_classification import IntelImageClassification
from .isic2016 import ISIC2016
from .k_mnist import KMNIST
from .k_mnist49 import KMNIST49
from .kannada_mnist import KannadaMNIST
from .kather_colorectal_histology import KatherColorectalHistology
from .kth_tips import KTHTIPS
from .linnaeus5 import Linnaeus5
from .malaria import Malaria
from .med_mnist import MedMNIST
from .mnist_c import MNISTC
from .mnist_m import MNISTM
from .not_mnist import NotMNIST

# from .mnist import MNIST
from .office_home import OfficeHome
from .olivetti_faces import OlivettiFaces
from .optdigits import Optdigits
from .oracle_mnist import OracleMNIST
from .oxford_iiit_pet import OxfordIIITPet
from .pacs import PACS
from .patch_camelyon import PatchCamelyon
from .patternnet import PatternNet
from .plant_village import PlantVillage
from .plantdoc import PlantDoc

# from .places365_small import Places365Small
from .rendered_sst2 import RenderedSST2
from .rock_paper_scissor import RockPaperScissor
from .scene8 import Scene8
from .semeion import Semeion
from .shapes3d import Shapes3D
from .small_norb import SmallNORB
from .stanford40 import Stanford40
from .stanford_dogs import StanfordDogs
from .stl10 import STL10
from .sun397 import SUN397
from .svhn import SVHN
from .tf_flowers import TFFlowers
from .tiny_imagenet import TinyImagenet
from .tiny_imagenet_c import TinyImagenetC
from .usps import USPS
from .weizmann_seg import WeizmannSegmentation


__all__ = [
    "AIDERv2",
    "ArabicCharacters",
    "ArabicDigits",
    "AWA2",
    "Beans",
    "Caltech101",
    "BeeDataset",
    "BelgiumTSC",
    "BreakHis",
    "BSDS500",
    "Caltech101Silhouettes",
    "Caltech256",
    "Cars196",
    "Cars3D",
    "CCT20",
    "CIFAR10",
    "CIFAR10_1",
    "CIFAR10_2",
    "CIFAR10C",
    "CIFAR100",
    "CIFAR100C",
    "CINIC10",
    "CLEVRER",
    "CMUFaces",
    "COIL20",
    "COIL100",
    "Country211",
    "CUB200",
    "Devanagari",
    "DeepWeeds",
    "DeFungi",
    "DSprites",
    "DSpritesColor",
    "DSpritesNoise",
    "DSpritesScream",
    "DTD",
    "EMNIST",
    "ETH80",
    "EuroSAT",
    "FacePointing",
    "FashionMNIST",
    "FGVCAircraft",
    "Flowers17",
    "Flowers102",
    "FMD",
    "Galaxy10Decal",
    "Food101",
    "GroceryStore",
    "FreiburgGroceries",
    "Fruits360",
    "GTSRB",
    "HASYv2",
    "HorseOrHuman",
    "ImageNet1K",
    "ImageNet100",
    "ImageNetSketch",
    "ImageNetV2",
    "ImageNetA",
    "ImageNetR",
    "Imagenette",
    "IntelImageClassification",
    "ISIC2016",
    "TFFlowers",
    "Imagewoof",
    "KMNIST",
    "KannadaMNIST",
    "KTHTIPS",
    "KMNIST49",
    "KatherColorectalHistology",
    "Linnaeus5",
    "Malaria",
    "MedMNIST",
    "MNISTM",
    "MNISTC",
    "NotMNIST",
    "PlantDoc",
    "OlivettiFaces",
    "Optdigits",
    "PlantVillage",
    "OfficeHome",
    "PACS",
    "PatternNet",
    "PatchCamelyon",
    "OxfordIIITPet",
    "OracleMNIST",
    "RenderedSST2",
    "RockPaperScissor",
    "Scene8",
    "Semeion",
    "Shapes3D",
    "SmallNORB",
    "Stanford40",
    "STL10",
    "StanfordDogs",
    "SUN397",
    "SVHN",
    "TinyImagenet",
    "TinyImagenetC",
    "USPS",
    "WeizmannSegmentation",
]
