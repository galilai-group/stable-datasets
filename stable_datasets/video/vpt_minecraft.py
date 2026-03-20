"""VPT Minecraft contractor demonstrations dataset.

The Video PreTraining (VPT) dataset contains Minecraft gameplay contractor
demonstrations consisting of synchronized video recordings (.mp4) and
action/state files (.jsonl). Each segment is up to 5 minutes long and
captures screen observations, actions, environment statistics, and a
checkpoint save file from the start of the segment.

Available recorder versions:
  - v6:         Core recorder (free play, general)
  - v7:         Prompt changes (emphasises early game / tree chopping)
  - v8:         House building from scratch (10 min, task-specific)
  - v9:         House building from random starting materials (10 min)
  - v10:        Obtain diamond pickaxe (20 min)
  - find_cave:  BASALT 2022 – find a cave (3 min)
  - waterfall:  BASALT 2022 – build a waterfall (5 min)
  - animal_pen: BASALT 2022 – build a village animal pen (5 min)
  - build_house: BASALT 2022 – build a village house (12 min)

Each segment provides:
  - relpath:         "<recorder-version>/<alias>-<session>-<date>-<time>"
  - video_url:       URL to the .mp4 video recording (720p captured → 360p/20hz)
  - action_url:      URL to the .jsonl action file (one action dict per line)
  - options_url:     URL to the -options.json recorder configuration
  - checkpoint_url:  URL to the .zip Minecraft world checkpoint

Note: Video files are large (tens of MB per segment). This class returns
URLs and metadata only; use ``load_actions()`` to fetch the lightweight
action JSONL for a segment.

License: MIT
Paper: https://cdn.openai.com/vpt/Paper.pdf
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from filelock import FileLock
from tqdm import tqdm


HOMEPAGE = "https://github.com/openai/Video-Pre-Training"
LICENSE = "MIT"
CITATION = """@article{baker2022video,
  title={Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos},
  author={Baker, Bowen and Akkaya, Ilge and Zhokhov, Peter and Huizinga, Joost and
          Tang, Jie and Ecoffet, Adrien and Houghton, Brandon and Sampedro, Raul and
          Clune, Jeff},
  journal={arXiv preprint arXiv:2206.11795},
  year={2022}
}"""

# Maps user-facing file-type names to the corresponding segment dict key.
_FILE_KEY_MAP: dict[str, str] = {
    "video": "video_url",
    "actions": "action_url",
    "options": "options_url",
    "checkpoint": "checkpoint_url",
}

# Index JSON files listing available segments per recorder version.
# Each JSON contains {"basedir": "...", "relpaths": [...]}.
INDEX_URLS: dict[str, str] = {
    "v6": "https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_6xx_Jun_29.json",
    "v7": "https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_7xx_Apr_6.json",
    "v8": "https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_8xx_Jun_29.json",
    "v9": "https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_9xx_Jun_29.json",
    "v10": "https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_10xx_Jun_29.json",
    "find_cave": "https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/find-cave-Jul-28.json",
    "waterfall": "https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/waterfall-Jul-28.json",
    "animal_pen": "https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/pen-animals-Jul-28.json",
    "build_house": "https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/build-house-Jul-28.json",
}


class VPTMinecraft:
    """VPT Minecraft contractor demonstrations.

    Provides access to OpenAI's Video PreTraining (VPT) contractor dataset —
    synchronized Minecraft gameplay video recordings (.mp4) paired with
    per-timestep action/state JSONL files.

    ``VPTMinecraft(version)`` fetches the index file for the requested recorder
    version from the OpenAI public mirror and returns a list of segment
    descriptor dicts.  Each dict contains the relpath and pre-computed URLs for
    the video, action, options, and checkpoint files.

    Examples:
        # Load segment list for the obtain-diamond-pickaxe version
        >>> segments = VPTMinecraft(version="v10")
        >>> print(len(segments))   # number of available segments
        >>> seg = segments[0]
        >>> print(seg["video_url"])

        # Fetch and inspect actions for one segment
        >>> actions = VPTMinecraft.load_actions(seg)
        >>> print(actions[0]["keyboard"]["keys"])

        # Iterate over the first 5 segments
        >>> for seg in VPTMinecraft.iter_segments(segments, max_segments=5):
        ...     print(seg["relpath"])

    Note:
        Video files are large (~50–100 MB per 5-minute segment). Only the
        action JSONL is fetched by ``load_actions()``; video and checkpoint
        files must be downloaded separately from the returned URLs.
    """

    HOMEPAGE = HOMEPAGE
    LICENSE = LICENSE
    CITATION = CITATION
    INDEX_URLS = INDEX_URLS

    def __new__(
        cls,
        version: str = "v10",
        **kwargs: Any,
    ) -> list[dict[str, str]]:
        """Fetch the index for the given version and return segment descriptors.

        Args:
            version: Recorder version to load. Must be one of:
                ``v6``, ``v7``, ``v8``, ``v9``, ``v10`` (main contractor data)
                or ``find_cave``, ``waterfall``, ``animal_pen``, ``build_house``
                (BASALT 2022 competition tasks).

        Returns:
            List of segment dicts, each containing:

            - ``relpath``:        Relative segment identifier
              (``<recorder-version>/<alias>-<session-id>-<date>-<time>``).
            - ``video_url``:      Full URL to the ``.mp4`` video recording.
            - ``action_url``:     Full URL to the ``.jsonl`` action file.
            - ``options_url``:    Full URL to the ``-options.json`` recorder config.
            - ``checkpoint_url``: Full URL to the ``.zip`` world checkpoint.

        Raises:
            ValueError: If ``version`` is not a recognised key.
            requests.HTTPError: If the index file cannot be fetched.
        """
        if version not in INDEX_URLS:
            raise ValueError(f"Unknown version '{version}'. Available versions: {list(INDEX_URLS.keys())}")

        index_url = INDEX_URLS[version]
        response = requests.get(index_url, timeout=30)
        response.raise_for_status()
        index = response.json()

        basedir = index["basedir"].rstrip("/")
        relpaths = index["relpaths"]

        segments = []
        for relpath in relpaths:
            # Some index files (e.g. BASALT) store relpaths with the file
            # extension already included; strip it so URLs aren't doubled.
            bare = relpath.removesuffix(".mp4").removesuffix(".jsonl")
            segments.append(
                {
                    "relpath": bare,
                    "video_url": f"{basedir}/{bare}.mp4",
                    "action_url": f"{basedir}/{bare}.jsonl",
                    "options_url": f"{basedir}/{bare}-options.json",
                    "checkpoint_url": f"{basedir}/{bare}.zip",
                }
            )
        return segments

    @classmethod
    def info(cls) -> dict[str, Any]:
        """Return dataset metadata."""
        return {
            "name": "VPTMinecraft",
            "homepage": cls.HOMEPAGE,
            "license": cls.LICENSE,
            "citation": cls.CITATION,
            "versions": list(cls.INDEX_URLS.keys()),
            "description": (
                "OpenAI VPT contractor demonstrations: synchronized Minecraft gameplay "
                "video recordings (.mp4) paired with per-timestep action/state files "
                "(.jsonl).  Available recorder versions: " + ", ".join(cls.INDEX_URLS.keys()) + "."
            ),
        }

    @staticmethod
    def load_actions(segment: dict[str, str]) -> list[dict[str, Any]]:
        """Fetch and parse the action JSONL file for a segment.

        Each line of the JSONL contains a single timestep's action dict with
        fields including ``mouse``, ``keyboard``, ``isGuiOpen``, ``hotbar``,
        ``yaw``, ``pitch``, ``xpos``/``ypos``/``zpos``, ``inventory``,
        ``stats``, and timing info (``tick``, ``milli``).

        Args:
            segment: A segment dict returned by ``VPTMinecraft()``.

        Returns:
            List of action dicts, one per recorded timestep (frame).

        Raises:
            requests.HTTPError: If the action file cannot be fetched.

        Example:
            >>> segments = VPTMinecraft(version="v10")
            >>> actions = VPTMinecraft.load_actions(segments[0])
            >>> print(actions[0]["keyboard"]["keys"])
            >>> print(actions[0]["mouse"]["dx"], actions[0]["mouse"]["dy"])
        """
        response = requests.get(segment["action_url"], timeout=60)
        response.raise_for_status()
        return [json.loads(line) for line in response.text.splitlines() if line.strip()]

    @staticmethod
    def iter_segments(
        segments: list[dict[str, str]],
        max_segments: int | None = None,
    ) -> Iterator[dict[str, str]]:
        """Iterate over segments with an optional limit.

        Args:
            segments: List of segment dicts returned by ``VPTMinecraft()``.
            max_segments: Maximum number of segments to yield.  ``None`` yields
                all segments.

        Yields:
            Segment dicts in order.

        Example:
            >>> segments = VPTMinecraft(version="v10")
            >>> for seg in VPTMinecraft.iter_segments(segments, max_segments=5):
            ...     print(seg["relpath"])
        """
        for i, seg in enumerate(segments):
            if max_segments is not None and i >= max_segments:
                break
            yield seg

    @staticmethod
    def fetch_video_chunk(segment: dict[str, str], num_bytes: int = 4096) -> bytes:
        """Fetch the first ``num_bytes`` bytes of a segment's video file.

        Uses an HTTP Range request so only a small slice is transferred,
        making it suitable for connectivity / availability checks without
        downloading the entire (potentially hundreds of MB) video.

        Args:
            segment:   A segment dict returned by ``VPTMinecraft()``.
            num_bytes: Number of bytes to fetch from the start of the file.
                Defaults to 4096 (4 KB).

        Returns:
            Raw bytes of the first ``num_bytes`` of the ``.mp4``.

        Raises:
            requests.HTTPError: If the server rejects the range request.

        Example:
            >>> segments = VPTMinecraft(version="v10")
            >>> chunk = VPTMinecraft.fetch_video_chunk(segments[0], num_bytes=1024)
            >>> assert chunk[:4] == b'\x00\x00\x00\x18'  # ftyp MP4 box
        """
        headers = {"Range": f"bytes=0-{num_bytes - 1}"}
        response = requests.get(segment["video_url"], headers=headers, timeout=30)
        response.raise_for_status()
        return response.content

    @staticmethod
    def download_segment(
        segment: dict[str, str],
        dest_dir: str | Path | None = None,
        files: tuple[str, ...] = ("video", "actions"),
        progress_bar: bool = True,
    ) -> dict[str, Path]:
        """Download files for a single segment, skipping any already on disk.

        Files are stored under::

            {dest_dir}/vpt_minecraft/{recorder-version}/{filename}

        Each file is written atomically (to a ``.tmp`` file, then renamed) and
        protected by a ``filelock`` so concurrent callers never corrupt the
        same file.

        Args:
            segment:      A segment dict returned by ``VPTMinecraft()``.
            dest_dir:     Root directory for downloads.  Defaults to
                          ``~/.stable-datasets/downloads/``.
            files:        Which file types to download.  Any subset of
                          ``"video"``, ``"actions"``, ``"options"``,
                          ``"checkpoint"``.  Defaults to
                          ``("video", "actions")``.
            progress_bar: Show a ``tqdm`` progress bar per file.
                          Defaults to ``True``.

        Returns:
            Dict mapping each requested file type to its local ``Path``.

        Raises:
            ValueError: If an unknown file type is requested.
            requests.HTTPError: If a download fails.

        Example:
            >>> segments = VPTMinecraft(version="v10")
            >>> paths = VPTMinecraft.download_segment(segments[0])
            >>> print(paths["video"])    # Path to the local .mp4
            >>> print(paths["actions"])  # Path to the local .jsonl
        """
        unknown = set(files) - set(_FILE_KEY_MAP)
        if unknown:
            raise ValueError(f"Unknown file types: {unknown}. Valid: {set(_FILE_KEY_MAP)}")

        if dest_dir is None:
            from stable_datasets.utils import _get_cache_dir

            dest_dir = Path(os.path.expanduser(_get_cache_dir())) / "downloads"
        dest_dir = Path(dest_dir)

        version = segment["relpath"].split("/")[0]
        local_paths: dict[str, Path] = {}

        for file_type in files:
            url = segment[_FILE_KEY_MAP[file_type]]
            filename = url.rsplit("/", 1)[-1]
            dest = dest_dir / "vpt_minecraft" / version / filename
            dest.parent.mkdir(parents=True, exist_ok=True)

            lock_path = dest.with_suffix(dest.suffix + ".lock")
            tmp_path = dest.with_suffix(dest.suffix + ".tmp")

            with FileLock(lock_path):
                if not dest.exists():
                    try:
                        with requests.get(url, stream=True, timeout=(10, 300)) as resp:
                            resp.raise_for_status()
                            total = int(resp.headers.get("content-length", 0) or 0)
                            with (
                                open(tmp_path, "wb") as f,
                                tqdm(
                                    desc=f"{file_type}: {filename}",
                                    total=total or None,
                                    unit="B",
                                    unit_scale=True,
                                    unit_divisor=1024,
                                    disable=not progress_bar,
                                ) as bar,
                            ):
                                for chunk in resp.iter_content(chunk_size=65536):
                                    if chunk:
                                        f.write(chunk)
                                        bar.update(len(chunk))
                        tmp_path.replace(dest)
                    except Exception:
                        if tmp_path.exists():
                            tmp_path.unlink()
                        raise

            local_paths[file_type] = dest

        return local_paths

    @staticmethod
    def download_segments(
        segments: list[dict[str, str]],
        dest_dir: str | Path | None = None,
        files: tuple[str, ...] = ("video", "actions"),
        max_segments: int | None = None,
        max_workers: int = 4,
    ) -> list[dict[str, Path]]:
        """Download files for multiple segments concurrently.

        Iterates over ``segments`` (optionally capped by ``max_segments``) and
        downloads the requested file types using a thread pool.
        Already-cached files are skipped automatically.

        Args:
            segments:     List of segment dicts returned by ``VPTMinecraft()``.
            dest_dir:     Root directory for downloads.  Defaults to
                          ``~/.stable-datasets/downloads/``.
            files:        Which file types to download per segment.  Any subset
                          of ``"video"``, ``"actions"``, ``"options"``,
                          ``"checkpoint"``.  Defaults to
                          ``("video", "actions")``.
            max_segments: Maximum number of segments to download.  ``None``
                          downloads all segments in the list.
            max_workers:  Number of concurrent download threads.  Defaults to 4.

        Returns:
            List of dicts (one per segment, in input order) mapping file type
            to its local ``Path``.

        Example:
            >>> segments = VPTMinecraft(version="v10")
            >>> all_paths = VPTMinecraft.download_segments(
            ...     segments, max_segments=10
            ... )
            >>> for paths in all_paths:
            ...     print(paths["video"], paths["actions"])
        """
        subset = list(VPTMinecraft.iter_segments(segments, max_segments=max_segments))
        results: list[dict[str, Path]] = [{}] * len(subset)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(VPTMinecraft.download_segment, seg, dest_dir, files, False): i
                for i, seg in enumerate(subset)
            }
            with tqdm(total=len(subset), desc="Downloading segments", unit="seg") as bar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    results[idx] = future.result()
                    bar.update(1)

        return results

    @staticmethod
    def parse_relpath(relpath: str) -> dict[str, str]:
        """Parse a segment relpath into its component parts.

        Relpaths follow the format:
        ``<recorder-version>/<contractor-alias>-<session-id>-<date>-<time>``

        where ``session-id`` is a 12-character hex string, ``date`` is
        ``YYYYMMDD``, and ``time`` is ``HHMMSS``.

        Args:
            relpath: Segment relpath, e.g.
                ``"10.0/cheeky-cornflower-setter-02e496ce4abb-20220421-092639"``.

        Returns:
            Dict with keys:

            - ``version``:           Recorder version string (e.g. ``"10.0"``).
            - ``contractor_alias``:  Contractor username (e.g.
              ``"cheeky-cornflower-setter"``).
            - ``session_id``:        12-char hex session identifier.
            - ``date``:              Recording date as ``"YYYYMMDD"``.
            - ``time``:              Recording start time as ``"HHMMSS"``.

        Example:
            >>> info = VPTMinecraft.parse_relpath(
            ...     "10.0/cheeky-cornflower-setter-02e496ce4abb-20220421-092639"
            ... )
            >>> info["version"]
            '10.0'
            >>> info["contractor_alias"]
            'cheeky-cornflower-setter'
        """
        version, filename = relpath.split("/", 1)
        # Split from right: last three separators give time, date, session_id;
        # everything before is the contractor alias (which may contain hyphens).
        parts = filename.rsplit("-", 3)
        if len(parts) == 4:
            alias, session_id, date, time_str = parts
        else:
            alias, session_id, date, time_str = filename, "", "", ""

        return {
            "version": version,
            "contractor_alias": alias,
            "session_id": session_id,
            "date": date,
            "time": time_str,
        }
