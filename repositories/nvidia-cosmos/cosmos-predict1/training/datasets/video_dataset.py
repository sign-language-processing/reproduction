# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Run this command to interactively debug:
PYTHONPATH=. python cosmos_predict1/tokenizer/training/datasets/video_dataset.py

Adapted from:
https://github.com/bytedance/IRASim/blob/main/dataset/dataset_3D.py
"""

import os
import time
import traceback
import warnings
from glob import glob

import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from torchvision import transforms as T

from cosmos_predict1.diffusion.utils.dataset_utils import ToTensorVideo


class Dataset(Dataset):
    def __init__(
        self,
        video_pattern,
        num_video_frames=25,
        **kwargs,  # absorb unused args from config system
    ):
        """Dataset class for loading video data with random frame sampling.

        Args:
            video_pattern (str): path/to/videos/*.mp4 or path/to/videos/**/*.mp4
            num_video_frames (int): Number of consecutive frames to load per sequence
        """

        super().__init__()
        self.video_directory_or_pattern = video_pattern
        self.sequence_length = num_video_frames

        self.video_paths = self._discover_videos(str(video_pattern))
        print(f"{len(self.video_paths)} videos found")

        self.wrong_number = 0
        self._bad_paths: set = self._load_bad_paths_cache(str(video_pattern))
        self.preprocess = T.Compose(
            [
                ToTensorVideo(),
            ]
        )

    @staticmethod
    def _base_dir(video_pattern):
        """Extract the stable base directory from a glob pattern."""
        return video_pattern.split('**')[0].split('*')[0].rstrip('/')

    @staticmethod
    def _bad_paths_cache_file(video_pattern):
        """Stable path for the persistent bad-paths cache file."""
        return os.path.join(Dataset._base_dir(video_pattern), "_bad_paths_cache.txt")

    @staticmethod
    def _load_bad_paths_cache(video_pattern):
        cache_file = Dataset._bad_paths_cache_file(video_pattern)
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                paths = {line.strip() for line in f if line.strip()}
            print(f"Loaded {len(paths)} known-bad paths from cache: {cache_file}")
            return paths
        return set()

    def _persist_bad_path(self, path):
        """Append a newly-discovered bad path to the on-disk cache (best-effort)."""
        try:
            cache_file = self._bad_paths_cache_file(str(self.video_directory_or_pattern))
            with open(cache_file, "a") as f:
                f.write(path + "\n")
        except OSError:
            pass  # read-only volume or race condition — in-memory cache still works

    @staticmethod
    def _size_prefilter(all_paths, bad_cache):
        """First-run filter: drop empty/missing files using parallel stat calls.

        Only runs once (when no bad_cache exists yet). Saves bad paths to the cache
        so subsequent runs subtract them via the normal glob-minus-cache path.
        """
        from concurrent.futures import ThreadPoolExecutor

        def is_readable(path):
            try:
                return os.path.getsize(path) > 0
            except OSError:
                return False

        print(f"[rank 0] Size-filtering {len(all_paths)} videos (first run, stat-only)...")
        t0 = time.time()

        valid, bad = [], []
        with ThreadPoolExecutor(max_workers=64) as exe:
            for path, ok in zip(all_paths, exe.map(is_readable, all_paths)):
                (valid if ok else bad).append(path)

        elapsed = time.time() - t0
        print(f"[rank 0] Size-filter done in {elapsed:.1f}s: {len(valid)} valid, {len(bad)} empty/missing")

        if bad:
            with open(bad_cache, 'a') as f:
                f.write('\n'.join(bad) + '\n')

        return valid

    @staticmethod
    def _discover_videos(video_pattern):
        """Discover video files with DDP-safe coordination.

        Every run: globs all files then subtracts the accumulated bad-paths cache,
        so the valid list is always fresh without a separate cache file to manage.
        First run (no bad-paths cache yet): runs a fast size-filter to seed the cache.

        Uses a sentinel file to signal completion — other ranks wait for the sentinel
        rather than polling the data file, eliminating sleep(2) race conditions.
        """
        rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
        bad_cache = Dataset._bad_paths_cache_file(video_pattern)
        ready_sentinel = "/tmp/_video_paths_ready"
        paths_cache = "/tmp/_video_paths.txt"

        if rank == 0:
            # Remove stale tmp files from previous runs
            for f in (ready_sentinel, paths_cache):
                try:
                    os.remove(f)
                except FileNotFoundError:
                    pass

            print(f"[rank 0] Globbing {video_pattern} ...")
            all_paths = sorted(glob(video_pattern, recursive=True))

            if os.path.exists(bad_cache):
                with open(bad_cache) as f:
                    known_bad = {line.strip() for line in f if line.strip()}
                paths = [p for p in all_paths if p not in known_bad]
                print(f"[rank 0] {len(all_paths)} found, {len(all_paths) - len(paths)} known-bad removed, {len(paths)} ready")
            else:
                paths = Dataset._size_prefilter(all_paths, bad_cache)

            # Write paths then sentinel (sentinel guarantees paths file is complete)
            with open(paths_cache, 'w') as f:
                f.write('\n'.join(paths))
            with open(ready_sentinel, 'w') as f:
                f.write(str(len(paths)))
            return paths
        else:
            # Wait for rank 0's sentinel (guaranteed complete when sentinel exists)
            deadline = time.monotonic() + 300  # 5-minute timeout
            while time.monotonic() < deadline:
                if os.path.exists(ready_sentinel):
                    with open(paths_cache) as f:
                        paths = [line.strip() for line in f if line.strip()]
                    print(f"[rank {rank}] Loaded {len(paths)} videos from cache")
                    return paths
                time.sleep(1)
            raise RuntimeError(f"[rank {rank}] Timeout waiting for video file list from rank 0")

    def __str__(self):
        return f"{len(self.video_paths)} samples from {self.video_directory_or_pattern}"

    def __len__(self):
        return len(self.video_paths)

    def _get_frames(self, video_path):
        # Open video and get number of frames
        vr = VideoReader(video_path, ctx=cpu(0))
        n_frames = len(vr)

        # Sample a random start frame, ensure enough frames for sequence
        max_start = n_frames - self.sequence_length
        if max_start < 0:
            raise ValueError(f"Video {video_path} has {n_frames} frames, need {self.sequence_length}")
        start_frame = np.random.randint(0, max_start + 1)
        frame_ids = list(range(start_frame, start_frame + self.sequence_length))

        # Load frames
        vr.seek(0)
        frame_data = vr.get_batch(frame_ids).asnumpy()
        frames = frame_data.astype(np.uint8)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (l, c, h, w)
        frames = self.preprocess(frames)
        frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)

        return frames, start_frame

    def __getitem__(self, index):
        for _ in range(50):  # max retries to avoid infinite recursion on corrupt datasets
            video_path = self.video_paths[index]

            # Skip known-bad files immediately
            if video_path in self._bad_paths:
                index = np.random.randint(len(self.video_paths))
                continue

            try:
                data = dict()

                video, start_frame = self._get_frames(video_path)
                video = video.permute(1, 0, 2, 3)  # Rearrange from [T, C, H, W] to [C, T, H, W]
                data["video"] = video
                data["video_name"] = {
                    "video_path": video_path,
                    "start_frame_id": str(start_frame),
                }
                data["fps"] = 24
                data["image_size"] = torch.tensor([704, 1280, 704, 1280])
                data["num_frames"] = self.sequence_length
                data["padding_mask"] = torch.zeros(1, 704, 1280)
                data["loss_mask"] = torch.ones_like(video, dtype=torch.float32)

                return data

            except Exception:
                self._bad_paths.add(video_path)
                self._persist_bad_path(video_path)
                warnings.warn(
                    f"Invalid data encountered: {video_path}. "
                    f"Skipped (by randomly sampling another sample)."
                )
                warnings.warn(traceback.format_exc())
                self.wrong_number += 1
                index = np.random.randint(len(self.video_paths))

        raise RuntimeError(f"Failed to load a valid video after 50 retries (bad_paths cache size: {len(self._bad_paths)})")


if __name__ == "__main__":
    dataset = Dataset(
        video_pattern="assets/example_training_data/videos/*.mp4",
        num_video_frames=57,
    )

    indices = [0, 13, 200, -1]
    for idx in indices:
        data = dataset[idx]
        print(f"{idx=} " f"{data['video'].sum()=}\n" f"{data['video'].shape=}\n" f"{data['video_name']=}\n" "---")