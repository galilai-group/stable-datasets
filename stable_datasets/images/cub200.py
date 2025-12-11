import tarfile
import datasets
from PIL import Image
import io
import pandas as pd
from pathlib import Path


class CUB200(datasets.GeneratorBasedBuilder):
    """Caltech-UCSD Birds-200-2011 (CUB-200-2011) Dataset"""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description="""The Caltech-UCSD Birds-200-2011 dataset consists of 11,788 images of 200 bird species.""",
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(names=self._labels())
                }
            ),
            supervised_keys=("image", "label"),
            homepage="https://www.vision.caltech.edu/datasets/cub_200_2011/",
            citation="""@techreport{WahCUB_200_2011,
                         Title = {The Caltech-UCSD Birds-200-2011 Dataset},
                         Author = {Wah, C. and Branson, S. and Welinder, P. and Perona, P. and Belongie, S.},
                         Year = {2011},
                         Institution = {California Institute of Technology},
                         Number = {CNS-TR-2011-001}}"""
        )

    def _split_generators(self, dl_manager):
        # Download and extract in a single step
        extracted_path = dl_manager.download_and_extract("https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1")
        data_dir = Path(extracted_path) / "CUB_200_2011"

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data_dir": data_dir, "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"data_dir": data_dir, "split": "test"},
            )
        ]

    def _generate_examples(self, data_dir, split):
        """Generate examples from the extracted directory."""
        # Paths to metadata files in the extracted directory
        image_labels_path = data_dir / "image_class_labels.txt"
        image_paths_path = data_dir / "images.txt"
        train_test_split_path = data_dir / "train_test_split.txt"

        # Load metadata
        images_df = pd.read_csv(image_paths_path, sep='\s+', header=None, names=["image_id", "file_path"])
        labels_df = pd.read_csv(image_labels_path, sep='\s+', header=None, names=["image_id", "label"])
        split_df = pd.read_csv(train_test_split_path, sep='\s+', header=None, names=["image_id", "is_training"])

        # Merge metadata into a single DataFrame
        data_df = images_df.merge(labels_df, on="image_id").merge(split_df, on="image_id")
        data_df["label"] -= 1  # Zero-index the labels

        # Filter by the specified split
        is_training_split = 1 if split == "train" else 0
        split_data = data_df[data_df["is_training"] == is_training_split]

        # Generate examples
        for _, row in split_data.iterrows():
            image_path = data_dir / "images" / row['file_path']
            label = row["label"]

            # Load the image
            with open(image_path, "rb") as img_file:
                image = Image.open(img_file).convert("RGB")
                yield row["image_id"], {
                    "image": image,
                    "label": label,
                }

    @staticmethod
    def _labels():
        return [
            "Black_footed_Albatross", "Laysan_Albatross", "Sooty_Albatross", "Groove_billed_Ani",
            "Crested_Auklet", "Least_Auklet", "Parakeet_Auklet", "Rhinoceros_Auklet", "Brewer_Blackbird",
            "Red_winged_Blackbird", "Rusty_Blackbird", "Yellow_headed_Blackbird", "Bobolink",
            "Indigo_Bunting", "Lazuli_Bunting", "Painted_Bunting", "Cardinal", "Spotted_Catbird",
            "Gray_Catbird", "Yellow_breasted_Chat", "Eastern_Towhee", "Chuck_will_Widow",
            "Brandt_Cormorant", "Red_faced_Cormorant", "Pelagic_Cormorant", "Bronzed_Cowbird",
            "Shiny_Cowbird", "Brown_Creeper", "American_Crow", "Fish_Crow", "Black_billed_Cuckoo",
            "Mangrove_Cuckoo", "Yellow_billed_Cuckoo", "Gray_crowned_Rosy_Finch", "Purple_Finch",
            "Northern_Flicker", "Acadian_Flycatcher", "Great_Crested_Flycatcher", "Least_Flycatcher",
            "Olive_sided_Flycatcher", "Scissor_tailed_Flycatcher", "Vermilion_Flycatcher",
            "Yellow_bellied_Flycatcher", "Frigatebird", "Northern_Fulmar", "Gadwall", "American_Goldfinch",
            "European_Goldfinch", "Boat_tailed_Grackle", "Eared_Grebe", "Horned_Grebe",
            "Pied_billed_Grebe", "Western_Grebe", "Blue_Grosbeak", "Evening_Grosbeak", "Pine_Grosbeak",
            "Rose_breasted_Grosbeak", "Pigeon_Guillemot", "California_Gull", "Glaucous_winged_Gull",
            "Heermann_Gull", "Herring_Gull", "Ivory_Gull", "Ring_billed_Gull", "Slaty_backed_Gull",
            "Western_Gull", "Anna_Hummingbird", "Ruby_throated_Hummingbird", "Rufous_Hummingbird",
            "Green_Violetear", "Long_tailed_Jaeger", "Pomarine_Jaeger", "Blue_Jay", "Florida_Jay",
            "Green_Jay", "Dark_eyed_Junco", "Tropical_Kingbird", "Gray_Kingbird", "Belted_Kingfisher",
            "Green_Kingfisher", "Pied_Kingfisher", "Ringed_Kingfisher", "White_breasted_Kingfisher",
            "Red_legged_Kittiwake", "Horned_Lark", "Pacific_Loon", "Mallard", "Western_Meadowlark",
            "Hooded_Merganser", "Red_breasted_Merganser", "Mockingbird", "Nighthawk", "Clark_Nutcracker",
            "White_breasted_Nuthatch", "Baltimore_Oriole", "Hooded_Oriole", "Orchard_Oriole",
            "Scott_Oriole", "Ovenbird", "Brown_Pelican", "White_Pelican", "Western_Wood_Pewee",
            "Sayornis", "American_Pipit", "Whip_poor_Will", "Horned_Puffin", "Common_Raven",
            "White_necked_Raven", "American_Redstart", "Geococcyx", "Loggerhead_Shrike",
            "Great_Grey_Shrike", "Baird_Sparrow", "Black_throated_Sparrow", "Brewer_Sparrow",
            "Chipping_Sparrow", "Clay_colored_Sparrow", "House_Sparrow", "Field_Sparrow",
            "Fox_Sparrow", "Grasshopper_Sparrow", "Harris_Sparrow", "Henslow_Sparrow",
            "Le_Conte_Sparrow", "Lincoln_Sparrow", "Nelson_Sharp_tailed_Sparrow", "Savannah_Sparrow",
            "Seaside_Sparrow", "Song_Sparrow", "Tree_Sparrow", "Vesper_Sparrow",
            "White_crowned_Sparrow", "White_throated_Sparrow", "Cape_Glossy_Starling",
            "Bank_Swallow", "Barn_Swallow", "Cliff_Swallow", "Tree_Swallow", "Scarlet_Tanager",
            "Summer_Tanager", "Arctic_Tern", "Black_Tern", "Caspian_Tern", "Common_Tern",
            "Elegant_Tern", "Forster_Tern", "Least_Tern", "Green_tailed_Towhee", "Brown_Thrasher",
            "Sage_Thrasher", "Black_capped_Vireo", "Blue_headed_Vireo", "Philadelphia_Vireo",
            "Red_eyed_Vireo", "Warbling_Vireo", "White_eyed_Vireo", "Yellow_throated_Vireo",
            "Bay_breasted_Warbler", "Black_and_white_Warbler", "Black_throated_Blue_Warbler",
            "Blue_winged_Warbler", "Canada_Warbler", "Cape_May_Warbler", "Cerulean_Warbler",
            "Chestnut_sided_Warbler", "Golden_winged_Warbler", "Hooded_Warbler", "Kentucky_Warbler",
            "Magnolia_Warbler", "Mourning_Warbler", "Myrtle_Warbler", "Nashville_Warbler",
            "Orange_crowned_Warbler", "Palm_Warbler", "Pine_Warbler", "Prairie_Warbler",
            "Prothonotary_Warbler", "Swainson_Warbler", "Tennessee_Warbler", "Wilson_Warbler",
            "Worm_eating_Warbler", "Yellow_Warbler", "Northern_Waterthrush", "Louisiana_Waterthrush",
            "Bohemian_Waxwing", "Cedar_Waxwing", "American_Three_toed_Woodpecker",
            "Pileated_Woodpecker", "Red_bellied_Woodpecker", "Red_cockaded_Woodpecker",
            "Red_headed_Woodpecker", "Downy_Woodpecker", "Bewick_Wren", "Cactus_Wren",
            "Carolina_Wren", "House_Wren", "Marsh_Wren", "Rock_Wren", "Winter_Wren",
            "Common_Yellowthroat"
        ]
