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
    ):
        """Dataset class for loading image-text-to-video generation data with random sampling.

        Args:
            video_pattern (str): path/to/videos/*.mp4
            num_video_frames (int): Number of consecutive frames to load per sequence

        Returns dict with:
            - video: RGB frames tensor [C,T,H,W]
            - video_name: Dict with video path and start frame metadata

        Note:
            Each call to __getitem__ samples a random start frame from the video,
            then loads num_video_frames consecutive frames. If the end frame exceeds
            the video length, the sequence is padded with zeros.
        """

        super().__init__()
        self.video_directory_or_pattern = video_pattern
        self.sequence_length = num_video_frames

        self.video_paths = sorted(glob(str(video_pattern)))
        print(f"{len(self.video_paths)} videos in total")

        self.wrong_number = 0
        self.preprocess = T.Compose(
            [
                ToTensorVideo(),
            ]
        )

    def __str__(self):
        return f"{len(self.video_paths)} samples from {self.video_directory_or_pattern}"

    def __len__(self):
        return len(self.video_paths)

    def _get_frames(self, video_path):
        # Open video and get number of frames
        vr = VideoReader(video_path, ctx=cpu(0))
        n_frames = len(vr)

        # Sample a random start frame
        start_frame = np.random.randint(0, n_frames)
        end_frame = start_frame + self.sequence_length

        # Get available frames
        available_frames = min(end_frame, n_frames) - start_frame
        frame_ids = list(range(start_frame, start_frame + available_frames))

        # Load frames
        vr.seek(0)
        frame_data = vr.get_batch(frame_ids).asnumpy()
        frames = frame_data.astype(np.uint8)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (l, c, h, w)
        frames = self.preprocess(frames)
        frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)

        # Pad with zeros if needed
        if len(frame_ids) < self.sequence_length:
            # Get shape from first frame
            _, c, h, w = frames.shape
            padding_frames = self.sequence_length - len(frame_ids)
            padding = torch.zeros(padding_frames, c, h, w, dtype=torch.uint8)
            frames = torch.cat([frames, padding], dim=0)

        return frames, start_frame

    def __getitem__(self, index):
        try:
            video_path = self.video_paths[index]
            data = dict()

            video, start_frame = self._get_frames(video_path)
            video = video.permute(1, 0, 2, 3)  # Rearrange from [T, C, H, W] to [C, T, H, W]
            data["video"] = video
            data["video_name"] = {
                "video_path": video_path,
                "start_frame_id": str(start_frame),
            }
            data["fps"] = 24
            data["image_size"] = torch.tensor([704, 1280, 704, 1280])  # .cuda()  # TODO: Does this matter?
            data["num_frames"] = self.sequence_length
            data["padding_mask"] = torch.zeros(1, 704, 1280)  # .cuda()

            return data
        except Exception:
            warnings.warn(
                f"Invalid data encountered: {self.video_paths[index]}. Skipped "
                f"(by randomly sampling another sample in the same dataset)."
            )
            warnings.warn("FULL TRACEBACK:")
            warnings.warn(traceback.format_exc())
            self.wrong_number += 1
            print(self.wrong_number)
            return self[np.random.randint(len(self.video_paths))]


if __name__ == "__main__":
    dataset = Dataset(
        video_pattern="assets/example_training_data/videos/*.mp4",
        num_video_frames=57,
    )

    indices = [0, 13, 200, -1]
    for idx in indices:
        data = dataset[idx]
        print(f"{idx=} " f"{data['video'].sum()=}\n" f"{data['video'].shape=}\n" f"{data['video_name']=}\n" "---")