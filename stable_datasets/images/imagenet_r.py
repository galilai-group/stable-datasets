"""ImageNet-R: renditions (art, cartoons, sketches, toys, ...) of 200 ImageNet classes."""

import io
import tarfile

from PIL import Image as PILImage

from stable_datasets.images._imagenet_wnids import WNID_TO_IDX
from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import BaseDatasetBuilder


# The 200 ImageNet classes used by ImageNet-R as (WordNet ID, readable name) pairs, copied
# from the README.txt shipped inside the official archive (underscores replaced by spaces).
# They are sorted by WordNet ID, which is the same order as the canonical ImageNet-1K list,
# so label i means entry i here.
_IMAGENET_R_CLASSES: list[tuple[str, str]] = [
    ("n01443537", "goldfish"),
    ("n01484850", "great white shark"),
    ("n01494475", "hammerhead"),
    ("n01498041", "stingray"),
    ("n01514859", "hen"),
    ("n01518878", "ostrich"),
    ("n01531178", "goldfinch"),
    ("n01534433", "junco"),
    ("n01614925", "bald eagle"),
    ("n01616318", "vulture"),
    ("n01630670", "newt"),
    ("n01632777", "axolotl"),
    ("n01644373", "tree frog"),
    ("n01677366", "iguana"),
    ("n01694178", "African chameleon"),
    ("n01748264", "cobra"),
    ("n01770393", "scorpion"),
    ("n01774750", "tarantula"),
    ("n01784675", "centipede"),
    ("n01806143", "peacock"),
    ("n01820546", "lorikeet"),
    ("n01833805", "hummingbird"),
    ("n01843383", "toucan"),
    ("n01847000", "duck"),
    ("n01855672", "goose"),
    ("n01860187", "black swan"),
    ("n01882714", "koala"),
    ("n01910747", "jellyfish"),
    ("n01944390", "snail"),
    ("n01983481", "lobster"),
    ("n01986214", "hermit crab"),
    ("n02007558", "flamingo"),
    ("n02009912", "american egret"),
    ("n02051845", "pelican"),
    ("n02056570", "king penguin"),
    ("n02066245", "grey whale"),
    ("n02071294", "killer whale"),
    ("n02077923", "sea lion"),
    ("n02085620", "chihuahua"),
    ("n02086240", "shih tzu"),
    ("n02088094", "afghan hound"),
    ("n02088238", "basset hound"),
    ("n02088364", "beagle"),
    ("n02088466", "bloodhound"),
    ("n02091032", "italian greyhound"),
    ("n02091134", "whippet"),
    ("n02092339", "weimaraner"),
    ("n02094433", "yorkshire terrier"),
    ("n02096585", "boston terrier"),
    ("n02097298", "scottish terrier"),
    ("n02098286", "west highland white terrier"),
    ("n02099601", "golden retriever"),
    ("n02099712", "labrador retriever"),
    ("n02102318", "cocker spaniels"),
    ("n02106030", "collie"),
    ("n02106166", "border collie"),
    ("n02106550", "rottweiler"),
    ("n02106662", "german shepherd dog"),
    ("n02108089", "boxer"),
    ("n02108915", "french bulldog"),
    ("n02109525", "saint bernard"),
    ("n02110185", "husky"),
    ("n02110341", "dalmatian"),
    ("n02110958", "pug"),
    ("n02112018", "pomeranian"),
    ("n02112137", "chow chow"),
    ("n02113023", "pembroke welsh corgi"),
    ("n02113624", "toy poodle"),
    ("n02113799", "standard poodle"),
    ("n02114367", "timber wolf"),
    ("n02117135", "hyena"),
    ("n02119022", "red fox"),
    ("n02123045", "tabby cat"),
    ("n02128385", "leopard"),
    ("n02128757", "snow leopard"),
    ("n02129165", "lion"),
    ("n02129604", "tiger"),
    ("n02130308", "cheetah"),
    ("n02134084", "polar bear"),
    ("n02138441", "meerkat"),
    ("n02165456", "ladybug"),
    ("n02190166", "fly"),
    ("n02206856", "bee"),
    ("n02219486", "ant"),
    ("n02226429", "grasshopper"),
    ("n02233338", "cockroach"),
    ("n02236044", "mantis"),
    ("n02268443", "dragonfly"),
    ("n02279972", "monarch butterfly"),
    ("n02317335", "starfish"),
    ("n02325366", "wood rabbit"),
    ("n02346627", "porcupine"),
    ("n02356798", "fox squirrel"),
    ("n02363005", "beaver"),
    ("n02364673", "guinea pig"),
    ("n02391049", "zebra"),
    ("n02395406", "pig"),
    ("n02398521", "hippopotamus"),
    ("n02410509", "bison"),
    ("n02423022", "gazelle"),
    ("n02437616", "llama"),
    ("n02445715", "skunk"),
    ("n02447366", "badger"),
    ("n02480495", "orangutan"),
    ("n02480855", "gorilla"),
    ("n02481823", "chimpanzee"),
    ("n02483362", "gibbon"),
    ("n02486410", "baboon"),
    ("n02510455", "panda"),
    ("n02526121", "eel"),
    ("n02607072", "clown fish"),
    ("n02655020", "puffer fish"),
    ("n02672831", "accordion"),
    ("n02701002", "ambulance"),
    ("n02749479", "assault rifle"),
    ("n02769748", "backpack"),
    ("n02793495", "barn"),
    ("n02797295", "wheelbarrow"),
    ("n02802426", "basketball"),
    ("n02808440", "bathtub"),
    ("n02814860", "lighthouse"),
    ("n02823750", "beer glass"),
    ("n02841315", "binoculars"),
    ("n02843684", "birdhouse"),
    ("n02883205", "bow tie"),
    ("n02906734", "broom"),
    ("n02909870", "bucket"),
    ("n02939185", "cauldron"),
    ("n02948072", "candle"),
    ("n02950826", "cannon"),
    ("n02951358", "canoe"),
    ("n02966193", "carousel"),
    ("n02980441", "castle"),
    ("n02992529", "mobile phone"),
    ("n03124170", "cowboy hat"),
    ("n03272010", "electric guitar"),
    ("n03345487", "fire engine"),
    ("n03372029", "flute"),
    ("n03424325", "gasmask"),
    ("n03452741", "grand piano"),
    ("n03467068", "guillotine"),
    ("n03481172", "hammer"),
    ("n03494278", "harmonica"),
    ("n03495258", "harp"),
    ("n03498962", "hatchet"),
    ("n03594945", "jeep"),
    ("n03602883", "joystick"),
    ("n03630383", "lab coat"),
    ("n03649909", "lawn mower"),
    ("n03676483", "lipstick"),
    ("n03710193", "mailbox"),
    ("n03773504", "missile"),
    ("n03775071", "mitten"),
    ("n03888257", "parachute"),
    ("n03930630", "pickup truck"),
    ("n03947888", "pirate ship"),
    ("n04086273", "revolver"),
    ("n04118538", "rugby ball"),
    ("n04133789", "sandal"),
    ("n04141076", "saxophone"),
    ("n04146614", "school bus"),
    ("n04147183", "schooner"),
    ("n04192698", "shield"),
    ("n04254680", "soccer ball"),
    ("n04266014", "space shuttle"),
    ("n04275548", "spider web"),
    ("n04310018", "steam locomotive"),
    ("n04325704", "scarf"),
    ("n04347754", "submarine"),
    ("n04389033", "tank"),
    ("n04409515", "tennis ball"),
    ("n04465501", "tractor"),
    ("n04487394", "trombone"),
    ("n04522168", "vase"),
    ("n04536866", "violin"),
    ("n04552348", "military aircraft"),
    ("n04591713", "wine bottle"),
    ("n07614500", "ice cream"),
    ("n07693725", "bagel"),
    ("n07695742", "pretzel"),
    ("n07697313", "cheeseburger"),
    ("n07697537", "hotdog"),
    ("n07714571", "cabbage"),
    ("n07714990", "broccoli"),
    ("n07718472", "cucumber"),
    ("n07720875", "bell pepper"),
    ("n07734744", "mushroom"),
    ("n07742313", "Granny Smith"),
    ("n07745940", "strawberry"),
    ("n07749582", "lemon"),
    ("n07753275", "pineapple"),
    ("n07753592", "banana"),
    ("n07768694", "pomegranate"),
    ("n07873807", "pizza"),
    ("n07880968", "burrito"),
    ("n07920052", "espresso"),
    ("n09472597", "volcano"),
    ("n09835506", "baseball player"),
    ("n10565667", "scuba diver"),
    ("n12267677", "acorn"),
]

IMAGENET_R_WNIDS: list[str] = [wnid for wnid, _ in _IMAGENET_R_CLASSES]
IMAGENET_R_CLASS_NAMES: list[str] = [name for _, name in _IMAGENET_R_CLASSES]

# IMAGENET_R_TO_IN1K[i] is the ImageNet-1K class index of ImageNet-R label i.
# Use it to evaluate a 1000-way ImageNet classifier: keep only these 200 logits.
IMAGENET_R_TO_IN1K: list[int] = [WNID_TO_IDX[wnid] for wnid in IMAGENET_R_WNIDS]

_WNID_TO_LABEL: dict[str, int] = {wnid: i for i, wnid in enumerate(IMAGENET_R_WNIDS)}


class ImageNetR(BaseDatasetBuilder):
    """ImageNet-R: 30,000 renditions (art, cartoons, sketches, toys, ...) of 200 ImageNet classes.

    ImageNet-R is an evaluation-only robustness benchmark, so it has a single ``test``
    split. Labels are integers in [0, 200): label ``i`` is the class
    ``IMAGENET_R_CLASS_NAMES[i]`` with WordNet ID ``IMAGENET_R_WNIDS[i]``, and
    ``IMAGENET_R_TO_IN1K[i]`` is its ImageNet-1K index.
    """

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://github.com/hendrycks/imagenet-r",
        assets={
            "test": DownloadInfo(
                url="https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar",
                checksum="sha256:18c6bf493b39a0d975d48e587437f562caab9c52ae6327dcfa9dd8eb54aa1b52",
            ),
        },
        citation="""@article{hendrycks2021many,
                      title={The Many Faces of Robustness: A Critical Analysis of Out-of-Distribution Generalization},
                      author={Dan Hendrycks and Steven Basart and Norman Mu and Saurav Kadavath and Frank Wang and Evan Dorundo and Rahul Desai and Tyler Zhu and Samyak Parajuli and Mike Guo and Dawn Song and Jacob Steinhardt and Justin Gilmer},
                      journal={ICCV},
                      year={2021}}""",
        license="MIT",
    )

    def _info(self):
        return DatasetInfo(
            description="""ImageNet-R contains 30,000 renditions of 200 ImageNet classes: art, cartoons,
                deviantart, graffiti, embroidery, graphics, origami, paintings, patterns, plastic objects,
                plush objects, sculptures, sketches, tattoos, toys, and video games. It is a test-only
                benchmark for robustness to changes in image style.""",
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=IMAGENET_R_CLASS_NAMES),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
            license=self.SOURCE["license"],
        )

    def _generate_examples(self, data_path, split):
        # Archive layout: imagenet-r/<wnid>/<style>_<n>.jpg (plus README.txt and folder entries).
        # The style in a file name is only the search query used to collect the image (a noisy
        # hint, per the official README), so the label comes only from the folder.
        with tarfile.open(data_path, "r:*") as archive:
            for member in archive:
                if not member.isfile() or not member.name.lower().endswith((".jpg", ".jpeg")):
                    continue

                wnid = member.name.split("/")[-2]
                label = _WNID_TO_LABEL[wnid]  # KeyError on an unexpected folder: fail loudly

                image_bytes = archive.extractfile(member).read()
                with PILImage.open(io.BytesIO(image_bytes)) as img:
                    if img.mode == "RGB":
                        image = image_bytes  # keep the original JPEG bytes
                    else:
                        # About 4% of ImageNet-R is grayscale or CMYK: convert it to RGB and drop the
                        # embedded color profile, which describes the old colors (not RGB). Some CMYK
                        # profiles are also too large for PIL to read back from the stored PNG.
                        image = img.convert("RGB")
                        image.info.pop("icc_profile", None)

                yield member.name, {"image": image, "label": label}
