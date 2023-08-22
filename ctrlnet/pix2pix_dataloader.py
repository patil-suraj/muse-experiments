#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from argparse import Namespace

import webdataset as wds
from torchvision import transforms
from torch.utils.data import default_collate


def get_dataloader(train_shards_path_or_url, num_train_examples, per_gpu_batch_size, global_batch_size, num_workers=8):
    # num_train_examples: 313,010
    num_batches = math.ceil(num_train_examples / global_batch_size)
    num_worker_batches = math.ceil(
        num_train_examples / (global_batch_size * num_workers)
    )  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size

    # Preprocessing the datasets.
    def preprocess_images(sample):
        image = sample["image"]
        control_image = sample["control_image"]

        image = transforms.ToTensor()(image)
        image = transforms.Normalize([0.5], [0.5])(image)
        control_image = transforms.ToTensor()(control_image)

        return {
            "image": image,
            "control_image": control_image,
            "orig_size": (1024., 1024.),
            "crop_coords": (0., 0.),
        }

    dataset = (
        wds.WebDataset(train_shards_path_or_url, resampled=True, handler=wds.warn_and_continue)
        .shuffle(690, handler=wds.warn_and_continue)
        .decode("pil", handler=wds.warn_and_continue)
        .rename(
            image="edited_image.jpg",
            control_image="original_image.jpg",
            text="edit_prompt.txt",
            handler=wds.warn_and_continue,
        )
        .map(preprocess_images, handler=wds.warn_and_continue)
        .to_tuple("image", "control_image", "text", "orig_size", "crop_coords")
        .batched(per_gpu_batch_size, partial=False, collation_fn=default_collate)
        .with_epoch(num_worker_batches)
    )

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return dataloader


if __name__ == "__main__":
    args = Namespace(
        dataset_path="pipe:aws s3 cp s3://muse-datasets/instructpix2pix-clip-filtered-wds/{000000..000062}.tar -",
        num_train_examples=313010,
        per_gpu_batch_size=8,
        global_batch_size=64,
        num_workers=4,
        center_crop=False,
        random_flip=True,
        resolution=256,
        original_image_column="original_image",
        edit_prompt_column="edit_prompt",
        edited_image_column="edited_image",
    )
    dataloader = get_dataloader(args)
    for sample in dataloader:
        print(sample.keys())
        print(sample["original_images"].shape)
        print(sample["edited_images"].shape)
        print(len(sample["edit_prompts"]))
        for s, c in zip(sample["original_sizes"], sample["crop_top_lefts"]):
            print(f"Original size: {s}, {type(s)}")
            print(f"Crop: {c}, {type(c)}")
        break
