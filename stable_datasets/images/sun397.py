import io
import tarfile
import zipfile
from pathlib import Path

from PIL import Image as PILImage

from stable_datasets.schema import (
    BuilderConfig,
    ClassLabel,
    DatasetInfo,
    DatasetSource,
    DownloadInfo,
    Features,
    Version,
)
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, _default_dest_folder, bulk_download


class SUN397(BaseDatasetBuilder):
    """
    SUN-397

    SUN-397 is a scene recognition dataset containing 108,754 images across 397 scene
    categories. The original dataset provides ten predefined train/test partitions,
    each containing 50 training images and 50 testing images per class. All ten
    partitions are included in this implementation. An additional `all` configuration
    provides all available images as a single training split, intended for self-
    supervised learning (SSL) pipelines that do not require predefined train/test
    partitions.

    The original download link provided on the official SUN dataset website
    (https://3dvision.princeton.edu/projects/2010/SUN/) is no longer functional.
    Therefore, a third-party upload was used to obtain the dataset. To verify the
    integrity of the downloaded archive, its MD5 checksum was compared against the
    original checksum published on the official dataset website. The checksums matched,
    indicating that the downloaded archive matches the original distribution.
    """

    VERSION = Version("1.0.0")

    DEFAULT_CONFIG_NAME = "all"

    BUILDER_CONFIGS = [
        BuilderConfig(name="all", description="All SUN-397 images as one training set (no test split)"),
        BuilderConfig(
            name="partition_01", description="Partition (1) with 50 training and 50 testing images per class"
        ),
        BuilderConfig(
            name="partition_02", description="Partition (2) with 50 training and 50 testing images per class"
        ),
        BuilderConfig(
            name="partition_03", description="Partition (3) with 50 training and 50 testing images per class"
        ),
        BuilderConfig(
            name="partition_04", description="Partition (4) with 50 training and 50 testing images per class"
        ),
        BuilderConfig(
            name="partition_05", description="Partition (5) with 50 training and 50 testing images per class"
        ),
        BuilderConfig(
            name="partition_06", description="Partition (6) with 50 training and 50 testing images per class"
        ),
        BuilderConfig(
            name="partition_07", description="Partition (7) with 50 training and 50 testing images per class"
        ),
        BuilderConfig(
            name="partition_08", description="Partition (8) with 50 training and 50 testing images per class"
        ),
        BuilderConfig(
            name="partition_09", description="Partition (9) with 50 training and 50 testing images per class"
        ),
        BuilderConfig(
            name="partition_10", description="Partition (10) with 50 training and 50 testing images per class"
        ),
    ]

    SOURCE = DatasetSource(
        homepage="https://3dvision.princeton.edu/projects/2010/SUN/",
        assets={
            "archive": DownloadInfo(
                url="https://huggingface.co/zhengli97/SUN397_Dataset/resolve/main/SUN397.tar.gz?download=true",
                checksum="md5:8ca2778205c41d23104230ba66911c7a",
                filename="SUN397.tar.gz",
            ),
            "partitions": DownloadInfo(
                url="https://3dvision.princeton.edu/projects/2010/SUN/download/Partitions.zip",
                checksum="md5:29a205c0a0129d21f36cbecfefe81881",
                filename="Partitions.zip",
            ),
        },
        citation="""@inproceedings{5539970,
                    title     = {SUN database: Large-scale scene recognition from abbey to zoo},
                    author    = {Xiao, Jianxiong and Hays, James and Ehinger, Krista A. and Oliva, Aude and Torralba, Antonio},
                    year      = {2010},
                    booktitle = {2010 IEEE Computer Society Conference on Computer Vision and Pattern Recognition},
                    volume    = {},
                    number    = {},
                    pages     = {3485--3492},
                    doi       = {10.1109/CVPR.2010.5539970}}""",
        license="Non-commercial research use",
    )

    def _info(self):
        """Returns information about the SUN-397 dataset"""
        return DatasetInfo(
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=self._labels()),
                }
            ),
            description="SUN-397 is a large scene understanding dataset with 397 categories, totaling 108,754 images.",
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
            license=self.SOURCE["license"],
        )

    def _split_generators(self):
        """Custom _split_generator that supports multiple dataset partitions."""
        source = self._source()
        assets = source["assets"]

        download_dir = getattr(self, "_raw_download_dir", None)
        if download_dir is None:
            download_dir = _default_dest_folder()

        # bulk download the files
        asset_keys = list(assets.keys())
        downloaded_paths = bulk_download([assets[key] for key in asset_keys], dest_folder=download_dir)
        key_to_path = dict(zip(asset_keys, downloaded_paths))
        archive_path = key_to_path["archive"]
        partitions_path = key_to_path["partitions"]

        # initialize return value
        config_name = self.config.name
        ret = [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "archive_path": archive_path,
                    "partitions_path": partitions_path,
                    "partition": config_name,
                    "split": "train",
                },
            )
        ]

        # if config isn't "all" return a test split as well
        if config_name != "all":
            ret.append(
                SplitGenerator(
                    name=Split.TEST,
                    gen_kwargs={
                        "archive_path": archive_path,
                        "partitions_path": partitions_path,
                        "partition": config_name,
                        "split": "test",
                    },
                ),
            )

        return ret

    def _candidate_splits(self) -> list:
        """Custom _candidate_splits to ensure proper cache lookup"""
        if self.config.name == "all":
            return [Split.TRAIN]

        return [Split.TRAIN, Split.TEST]

    def _get_partition_images(self, partitions_path, partition, split):
        """Generate the set of images in this partition and split"""

        # lowercase inputs
        partition = partition.lower()
        split = split.lower()

        # check valid input
        valid_partitions = {builder_config.name for builder_config in self.BUILDER_CONFIGS}
        if partition not in valid_partitions:
            raise ValueError(f"received unknown data partition: {partition}. Must be one of: {valid_partitions}")
        if split not in ("train", "test"):
            raise ValueError(f"received unknown split type: {split}")

        # return empty set if no partition
        if partition == "all":
            return set()

        # get partition number from "partition_xx"
        part_num = partition.split("_")[1]

        # get partition file name
        part_file = f"{'Training' if split == 'train' else 'Testing'}_{part_num}.txt"

        # open partition file and get image list
        with zipfile.ZipFile(partitions_path, "r") as zipf:
            with zipf.open(part_file, "r") as partf:
                partition_images = {f"SUN397{line.decode().strip()}" for line in partf.readlines()}
        return partition_images

    def _generate_examples(self, archive_path, partitions_path, partition, split):
        """Generate examples from the .tar.gz archive."""

        # get image partition (if applicable)
        partition_images = self._get_partition_images(
            partitions_path=partitions_path,
            partition=partition,
            split=split,
        )

        # get the image files
        with tarfile.open(archive_path, "r:gz") as tar:
            # create tar member generator
            if partition_images:
                # only gets specified images
                members = (member for member in tar if member.isfile() and member.name in partition_images)
            else:
                # gets all available image files
                members = (member for member in tar if member.isfile() and Path(member.name).suffix == ".jpg")

            # generate examples from members and yield one by one
            for member in members:
                # check member type
                path = Path(member.name)
                if not member.isfile() or path.suffix != ".jpg":
                    raise ValueError(f"image path {member.name} is not an image file")

                # get label from image path (e.g. SUN397/a/abbey/sun_anipjfgcdipdbaur.jpg)
                # label may contain '/', so can't use path.parent.stem
                label = str(path.parent)[9:]

                # read image bytes
                with tar.extractfile(member) as f:
                    image_bytes = f.read()

                # ensure all images are RGB (lazy loading, only decodes if necessary)
                pil_image = PILImage.open(io.BytesIO(image_bytes))
                if pil_image.mode != "RGB":
                    image_bytes = ImageFeature().encode(pil_image.convert("RGB"))

                # yield training example, using path as key for traceability
                yield str(path), {"image": image_bytes, "label": label}

    @staticmethod
    def _labels():
        """Returns the list of SUN-397 labels (with the prefix removed)"""
        return [
            "abbey",
            "airplane_cabin",
            "airport_terminal",
            "alley",
            "amphitheater",
            "amusement_arcade",
            "amusement_park",
            "anechoic_chamber",
            "apartment_building/outdoor",
            "apse/indoor",
            "aquarium",
            "aqueduct",
            "arch",
            "archive",
            "arrival_gate/outdoor",
            "art_gallery",
            "art_school",
            "art_studio",
            "assembly_line",
            "athletic_field/outdoor",
            "atrium/public",
            "attic",
            "auditorium",
            "auto_factory",
            "badlands",
            "badminton_court/indoor",
            "baggage_claim",
            "bakery/shop",
            "balcony/exterior",
            "balcony/interior",
            "ball_pit",
            "ballroom",
            "bamboo_forest",
            "banquet_hall",
            "bar",
            "barn",
            "barndoor",
            "baseball_field",
            "basement",
            "basilica",
            "basketball_court/outdoor",
            "bathroom",
            "batters_box",
            "bayou",
            "bazaar/indoor",
            "bazaar/outdoor",
            "beach",
            "beauty_salon",
            "bedroom",
            "berth",
            "biology_laboratory",
            "bistro/indoor",
            "boardwalk",
            "boat_deck",
            "boathouse",
            "bookstore",
            "booth/indoor",
            "botanical_garden",
            "bow_window/indoor",
            "bow_window/outdoor",
            "bowling_alley",
            "boxing_ring",
            "brewery/indoor",
            "bridge",
            "building_facade",
            "bullring",
            "burial_chamber",
            "bus_interior",
            "butchers_shop",
            "butte",
            "cabin/outdoor",
            "cafeteria",
            "campsite",
            "campus",
            "canal/natural",
            "canal/urban",
            "candy_store",
            "canyon",
            "car_interior/backseat",
            "car_interior/frontseat",
            "carrousel",
            "casino/indoor",
            "castle",
            "catacomb",
            "cathedral/indoor",
            "cathedral/outdoor",
            "cavern/indoor",
            "cemetery",
            "chalet",
            "cheese_factory",
            "chemistry_lab",
            "chicken_coop/indoor",
            "chicken_coop/outdoor",
            "childs_room",
            "church/indoor",
            "church/outdoor",
            "classroom",
            "clean_room",
            "cliff",
            "cloister/indoor",
            "closet",
            "clothing_store",
            "coast",
            "cockpit",
            "coffee_shop",
            "computer_room",
            "conference_center",
            "conference_room",
            "construction_site",
            "control_room",
            "control_tower/outdoor",
            "corn_field",
            "corral",
            "corridor",
            "cottage_garden",
            "courthouse",
            "courtroom",
            "courtyard",
            "covered_bridge/exterior",
            "creek",
            "crevasse",
            "crosswalk",
            "cubicle/office",
            "dam",
            "delicatessen",
            "dentists_office",
            "desert/sand",
            "desert/vegetation",
            "diner/indoor",
            "diner/outdoor",
            "dinette/home",
            "dinette/vehicle",
            "dining_car",
            "dining_room",
            "discotheque",
            "dock",
            "doorway/outdoor",
            "dorm_room",
            "driveway",
            "driving_range/outdoor",
            "drugstore",
            "electrical_substation",
            "elevator/door",
            "elevator/interior",
            "elevator_shaft",
            "engine_room",
            "escalator/indoor",
            "excavation",
            "factory/indoor",
            "fairway",
            "fastfood_restaurant",
            "field/cultivated",
            "field/wild",
            "fire_escape",
            "fire_station",
            "firing_range/indoor",
            "fishpond",
            "florist_shop/indoor",
            "food_court",
            "forest/broadleaf",
            "forest/needleleaf",
            "forest_path",
            "forest_road",
            "formal_garden",
            "fountain",
            "galley",
            "game_room",
            "garage/indoor",
            "garbage_dump",
            "gas_station",
            "gazebo/exterior",
            "general_store/indoor",
            "general_store/outdoor",
            "gift_shop",
            "golf_course",
            "greenhouse/indoor",
            "greenhouse/outdoor",
            "gymnasium/indoor",
            "hangar/indoor",
            "hangar/outdoor",
            "harbor",
            "hayfield",
            "heliport",
            "herb_garden",
            "highway",
            "hill",
            "home_office",
            "hospital",
            "hospital_room",
            "hot_spring",
            "hot_tub/outdoor",
            "hotel/outdoor",
            "hotel_room",
            "house",
            "hunting_lodge/outdoor",
            "ice_cream_parlor",
            "ice_floe",
            "ice_shelf",
            "ice_skating_rink/indoor",
            "ice_skating_rink/outdoor",
            "iceberg",
            "igloo",
            "industrial_area",
            "inn/outdoor",
            "islet",
            "jacuzzi/indoor",
            "jail/indoor",
            "jail_cell",
            "jewelry_shop",
            "kasbah",
            "kennel/indoor",
            "kennel/outdoor",
            "kindergarden_classroom",
            "kitchen",
            "kitchenette",
            "labyrinth/outdoor",
            "lake/natural",
            "landfill",
            "landing_deck",
            "laundromat",
            "lecture_room",
            "library/indoor",
            "library/outdoor",
            "lido_deck/outdoor",
            "lift_bridge",
            "lighthouse",
            "limousine_interior",
            "living_room",
            "lobby",
            "lock_chamber",
            "locker_room",
            "mansion",
            "manufactured_home",
            "market/indoor",
            "market/outdoor",
            "marsh",
            "martial_arts_gym",
            "mausoleum",
            "medina",
            "moat/water",
            "monastery/outdoor",
            "mosque/indoor",
            "mosque/outdoor",
            "motel",
            "mountain",
            "mountain_snowy",
            "movie_theater/indoor",
            "museum/indoor",
            "music_store",
            "music_studio",
            "nuclear_power_plant/outdoor",
            "nursery",
            "oast_house",
            "observatory/outdoor",
            "ocean",
            "office",
            "office_building",
            "oil_refinery/outdoor",
            "oilrig",
            "operating_room",
            "orchard",
            "outhouse/outdoor",
            "pagoda",
            "palace",
            "pantry",
            "park",
            "parking_garage/indoor",
            "parking_garage/outdoor",
            "parking_lot",
            "parlor",
            "pasture",
            "patio",
            "pavilion",
            "pharmacy",
            "phone_booth",
            "physics_laboratory",
            "picnic_area",
            "pilothouse/indoor",
            "planetarium/outdoor",
            "playground",
            "playroom",
            "plaza",
            "podium/indoor",
            "podium/outdoor",
            "pond",
            "poolroom/establishment",
            "poolroom/home",
            "power_plant/outdoor",
            "promenade_deck",
            "pub/indoor",
            "pulpit",
            "putting_green",
            "racecourse",
            "raceway",
            "raft",
            "railroad_track",
            "rainforest",
            "reception",
            "recreation_room",
            "residential_neighborhood",
            "restaurant",
            "restaurant_kitchen",
            "restaurant_patio",
            "rice_paddy",
            "riding_arena",
            "river",
            "rock_arch",
            "rope_bridge",
            "ruin",
            "runway",
            "sandbar",
            "sandbox",
            "sauna",
            "schoolhouse",
            "sea_cliff",
            "server_room",
            "shed",
            "shoe_shop",
            "shopfront",
            "shopping_mall/indoor",
            "shower",
            "skatepark",
            "ski_lodge",
            "ski_resort",
            "ski_slope",
            "sky",
            "skyscraper",
            "slum",
            "snowfield",
            "squash_court",
            "stable",
            "stadium/baseball",
            "stadium/football",
            "stage/indoor",
            "staircase",
            "street",
            "subway_interior",
            "subway_station/platform",
            "supermarket",
            "sushi_bar",
            "swamp",
            "swimming_pool/indoor",
            "swimming_pool/outdoor",
            "synagogue/indoor",
            "synagogue/outdoor",
            "television_studio",
            "temple/east_asia",
            "temple/south_asia",
            "tennis_court/indoor",
            "tennis_court/outdoor",
            "tent/outdoor",
            "theater/indoor_procenium",
            "theater/indoor_seats",
            "thriftshop",
            "throne_room",
            "ticket_booth",
            "toll_plaza",
            "topiary_garden",
            "tower",
            "toyshop",
            "track/outdoor",
            "train_railway",
            "train_station/platform",
            "tree_farm",
            "tree_house",
            "trench",
            "underwater/coral_reef",
            "utility_room",
            "valley",
            "van_interior",
            "vegetable_garden",
            "veranda",
            "veterinarians_office",
            "viaduct",
            "videostore",
            "village",
            "vineyard",
            "volcano",
            "volleyball_court/indoor",
            "volleyball_court/outdoor",
            "waiting_room",
            "warehouse/indoor",
            "water_tower",
            "waterfall/block",
            "waterfall/fan",
            "waterfall/plunge",
            "watering_hole",
            "wave",
            "wet_bar",
            "wheat_field",
            "wind_farm",
            "windmill",
            "wine_cellar/barrel_storage",
            "wine_cellar/bottle_storage",
            "wrestling_ring/indoor",
            "yard",
            "youth_hostel",
        ]
