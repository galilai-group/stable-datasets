# Python Standard Library imports
import csv
import hashlib
import os
from functools import partial

# Third-party imports
import asyncio
import aiohttp
import datasets
from loguru import logger as logging
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from urllib.parse import urlparse

# Local imports
from stable_datasets.utils import _default_dest_folder, BaseDatasetBuilder


# Constants for downloading images
DOWNLOAD_BATCH_SIZE = 65536
FIRST_N_IMAGES_PER_SPLIT = 0  # 0 takes all the images in the split
LOG_FAILURES = False


def get_unique_filename(url: str) -> str:
    url_hash = hashlib.sha256(url.encode()).hexdigest()
    original_filename = os.path.basename(urlparse(url).path)
    ext = os.path.splitext(original_filename)[1] if '.' in original_filename else ''
    return f"{url_hash}{ext}"


def get_filepath(url: str, dest_folder: Path) -> Path:
    filename = get_unique_filename(url)
    return dest_folder / filename


async def safe_download(url, dest_folder, session, log_failure=False) -> bool:
    # Makes sure that the destination folder exists
    dest_folder.mkdir(parents=True, exist_ok=True)
    
    dest_path = get_filepath(url, dest_folder)
    try:
        async with session.get(url) as response:
            if response.status == 200:
                dest_path.write_bytes(await response.read())
                return True
    except asyncio.CancelledError:
        if log_failure:
            logging.warning(f"Received `asyncio.CancelledError` while downloading {url}")

        # In case the download was interrupted while writing bytes to the file,
        # remove the file so that the download isn't seen as successfully finished
        # later on
        if os.path.isfile(dest_path):
            os.remove(dest_path)
    except Exception as e:
        if log_failure:
            logging.warning(f"Failed to download {url}: {e}")
    return False


async def safe_bulk_download(urls, dest_folder, concurrency=100, timeout_seconds=15, log_failures=False):
    # Makes sure that the destination folder exists
    dest_folder.mkdir(parents=True, exist_ok=True)

    # Creates a client session with a timeout and a concurrency limit
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
    ) as session:
        # Limits the number of concurrent downloads by using a semaphore
        semaphore = asyncio.Semaphore(concurrency)
        async def sem_download(url):
            async with semaphore:
                return await safe_download(url, dest_folder, session, log_failure=log_failures)

        # Creates the list of tasks to download the URLs
        tasks = [asyncio.create_task(sem_download(url)) for url in urls]

        # Waits for the tasks to complete and returns the results
        results = []
        with tqdm(total=len(tasks), desc="Downloading images") as pbar:
            for coroutine in asyncio.as_completed(tasks):
                result = await coroutine
                results.append(result)
                pbar.update(1)
        return results


class CC3M(BaseDatasetBuilder):
    """CC3M (Conceptual Captions)"""

    VERSION = datasets.Version("1.0.0")

    SOURCE = {
        "homepage": "https://ai.google.com/research/ConceptualCaptions/download",
        "assets": {
            "train": "https://storage.googleapis.com/gcc-data/Train/GCC-training.tsv",
            "val": "https://storage.googleapis.com/gcc-data/Validation/GCC-1.1.0-Validation.tsv",
        },
        "citation": """@inproceedings{sharma-etal-2018-conceptual,
                        title = "Conceptual Captions: A Cleaned, Hypernymed, Image Alt-text Dataset For Automatic Image Captioning",
                        author = "Sharma, Piyush  and
                        Ding, Nan  and
                        Goodman, Sebastian  and
                        Soricut, Radu",
                        editor = "Gurevych, Iryna  and
                        Miyao, Yusuke",
                        booktitle = "Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
                        month = jul,
                        year = "2018",
                        address = "Melbourne, Australia",
                        publisher = "Association for Computational Linguistics",
                        url = "https://aclanthology.org/P18-1238/",
                        doi = "10.18653/v1/P18-1238",
                        pages = "2556--2565",
                    }""",
    }

    def _info(self):
        return datasets.DatasetInfo(
            description="""Conceptual Captions, colloquially known as CC3M, is a dataset of 3 million images with their corresponding captions constructed by a purely automated pipeline developed at Google AI. It is a large-scale dataset typically used for image captioning.""",
            features=datasets.Features(
                {"image": datasets.Image(), "caption": datasets.Value("string")}
            ),
            supervised_keys=("image", "caption"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        # Creates a subfolder for this split's images
        download_dir = getattr(self, "_raw_download_dir", _default_dest_folder())
        images_dir = Path(download_dir) / f"cc3m_{split}_images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Computes total number of images to download
        total_lines = sum(1 for _ in open(data_path))
        assert total_lines != 0, f"No {split} images found in {data_path}"
        logging.info(f"Total number of {split} images: {total_lines}")
        if FIRST_N_IMAGES_PER_SPLIT != 0 and FIRST_N_IMAGES_PER_SPLIT < total_lines:
            total_lines = FIRST_N_IMAGES_PER_SPLIT
            logging.info(f"Using only the first {total_lines} images from {split} split.")
        else:
            logging.info(f"Using all {total_lines} images from {split} split.")

        # Computes batch info
        num_batches = (total_lines + DOWNLOAD_BATCH_SIZE - 1) // DOWNLOAD_BATCH_SIZE
        last_batch_size = total_lines % DOWNLOAD_BATCH_SIZE
        if last_batch_size == 0:
            last_batch_size = DOWNLOAD_BATCH_SIZE
        logging.info(f"Total number of {split} download batches: {num_batches}")
        logging.info(f"Size of last download batch: {last_batch_size}")

        # First pass: downloads all images in batches, where the images in
        # each batch are downloaded in parallel
        successful_downloads = 0
        skipped_downloads = 0
        failed_downloads = 0
        with open(data_path) as f:
            reader = csv.reader(f, delimiter='\t')
            for batch_idx in tqdm(range(num_batches), desc=f"Processing {split} download batches"):
                # Finds batch size
                if batch_idx == num_batches - 1:
                    curr_batch_size = last_batch_size
                else:
                    curr_batch_size = DOWNLOAD_BATCH_SIZE

                # Gets all URLs for this batch by iterating through the rows of the file
                urls_to_download = []
                for _ in tqdm(range(curr_batch_size), desc=f"Getting URLs for batch {batch_idx} of split {split}"):
                    # Gets row
                    row = next(reader)
                    caption, image_url = row

                    # Checks if image file already exists
                    try:
                        if get_filepath(image_url, images_dir).is_file():
                            skipped_downloads += 1
                            continue
                    except OSError as e:
                        if e.errno == 36:  # "File name too long" to check if the path is a file
                            skipped_downloads += 1
                            continue
                        else:
                            raise e

                    # Adds URL to batch
                    urls_to_download.append(image_url)

                # Downloads batch
                logging.info(
                    f"In batch {batch_idx} of split {split}, "
                    f"there are {len(urls_to_download)} non-skipped images to download "
                    f"and {curr_batch_size - len(urls_to_download)} already-downloaded images"
                )
                results = asyncio.run(
                    safe_bulk_download(
                        urls_to_download,
                        dest_folder=images_dir,
                        log_failures=LOG_FAILURES,
                    )
                )
                num_succeeded = sum(results)
                successful_downloads += num_succeeded
                failed_downloads += len(results) - num_succeeded
        logging.info(f"Successfully downloaded {successful_downloads} {split} images.")
        logging.info(f"Failed downloads: {failed_downloads}")
        logging.info(f"Skipped downloads: {skipped_downloads}")

        # Second pass: opens and yields examples
        example_idx = 0
        invalid_images = 0
        with open(data_path) as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:  # Re-reads file instead of using `results` to simplify logic
                caption, image_url = row

                # Skips if the image wasn't downloaded
                image_path = get_filepath(image_url, images_dir)
                try:
                    if not image_path.is_file():
                        continue
                except OSError as e:
                    if e.errno == 36:  # "File name too long" to check if the path is a file
                        continue
                    else:
                        raise e
                
                # Tries to open and validate the image
                try:
                    # First verify the image is valid
                    with Image.open(image_path) as img_check:
                        img_check.verify()
                    # Then open it again for actual use (verify() renders the object unusable)
                    image = Image.open(image_path)
                except Exception as e:
                    if LOG_FAILURES:
                        logging.warning(f"Failed to open {image_path}: {e}")
                    invalid_images += 1
                    continue

                # Yields the example
                yield example_idx, {"image": image, "caption": caption}
                example_idx += 1

                # It's possible that more images than FIRST_N_IMAGES_PER_SPLIT
                # have been downloaded in total due to previous downloads, so
                # we check for that explicitly
                if example_idx >= total_lines:
                    break
        logging.info(f"Successfully generated {example_idx} examples from {split} split.")
        logging.info(f"Invalid images (couldn't be opened): {invalid_images}")
