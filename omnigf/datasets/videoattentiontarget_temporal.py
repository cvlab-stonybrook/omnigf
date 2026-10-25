# coding=utf-8

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

import os
from typing import Tuple, Union, Dict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import lightning.pytorch as pl
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from omnigf.utils import (
    generate_gaze_heatmap,
    generate_mask,
    square_bbox,
    Stage,
    pair,
)
from omnigf.utils.social_gaze import get_lah_labels, get_shuffle_idx
from omnigf.train.transforms import (
    RandomHeadBboxJitter,
    RandomHorizontalFlip,
    RandomCropSafeGaze,
    ColorJitter,
    Normalize,
    ToTensor,
    Compose,
    Resize,
)


TRAIN_SHOWS = [
    "Sherlock",
    "Hearing",
    "Modern_Family",
    "Cheers",
    "Star_Wars",
    "Veep",
    "BTS_at_Jimmy_Fallon",
    "Coveted",
    "Breaking_Bad",
    "Sound_of_Music",
    "Tartuffe",
    "Suits",
    "Driving_Miss_Daisy",
    "Crazy_Rich_Asian",
    "Keeping_Up_With_the_Kardashians",
    "Interview_at_the_Oscars",
    "Interview_with_Bill_Gates",
    "Arrested_Development",
    "A_Play_With_Words",
    "How_I_Met_Your_Mother",
    "Jersey_Shore",
    "My_Dinner_with_Andre",
    "Conan",
    "Band_of_Brothers",
    "The_View",
    "Seinfeld",
    "Grey's_Anatomy",
    "UFC_Octagon_Interview",
    "The_Ellen_Show",
    "Secret",
    "Friends",
    "Gone_with_the_Wind",
    "Three_Idiots",
    "All_in_the_Family",
    "Big_Bang_Theory",
    "Silicon_Valley",
    "Give_Me_One_Reason",
]
VAL_SHOWS = ["Orange_is_the_New_Black", "Before_Sunrise", "Project_Runway"]
TEST_SHOWS = [
    "CBS_This_Morning",
    "Downton_Abby",
    "Hell's_Kitchen",
    "I_Wanna_Marry_Harry",
    "It's_Always_Sunny_in_Philadelphia",
    "Jamie_Oliver",
    "MLB_Interview",
    "Survivor",
    "Titanic",
    "West_World",
]


class VideoAttentionTargetDataset_temporal(Dataset):
    def __init__(
        self,
        root: str,
        ann_root: str,
        split: str = "train",
        stride: int = 3,
        transform: Union[Compose, None] = None,
        tr: Tuple[float, float] = (-0.1, 0.1),
        image_size: Tuple[int, int] = (224, 224),
        head_size: Tuple[int, int] = (224, 224),
        heatmap_sigma: int = 3,
        heatmap_size: int = 64,
        num_people: int = 5,
        temporal_stride: int = 3,
        temporal_context: int = 2,
        aspect: bool = False,
    ):
        super().__init__()
        self.root = root
        self.ann_root = ann_root
        self.split = split
        self.stride = stride
        self.jitter_bbox = RandomHeadBboxJitter(p=1.0, tr=tr)
        self.transform = transform
        self.image_size = image_size
        self.head_size = head_size
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_size = heatmap_size
        self.num_people = num_people
        self.temporal_stride = temporal_stride
        self.temporal_context = temporal_context
        self.aspect = aspect

        # load annotations
        self.annotations = pd.read_hdf(
            os.path.join(ann_root, f"vat_{self.split}.h5"), "data"
        )
        self.annotations = self.annotations.groupby("path")
        self.paths = list(self.annotations.groups.keys())
        self.paths = np.array(self.paths)
        if self.stride > 1:
            index_keep = np.arange(len(self.paths), step=self.stride)
            self.paths = self.paths[index_keep]

    def __getitem__(self, index):
        path = self.paths[index]
        clip = "/".join(path.split("/")[:-1])
        frame = int(path.split("/")[-1][:-4])
        # jitter frame number during training
        if self.split == "train":
            if self.temporal_context == 0:
                frame_shift = torch.randint(
                    -(self.temporal_stride // 2), self.temporal_stride // 2, (1,)
                ).item()
            else:
                frame_shift = torch.randint(
                    -(self.temporal_context * self.temporal_stride - 1),
                    self.temporal_context * self.temporal_stride,
                    (1,),
                ).item()
            frame_tmp = frame + frame_shift
            path = os.path.join(clip, str(frame_tmp).zfill(8) + ".jpg")
            while path not in self.annotations.groups.keys():
                if self.temporal_context == 0:
                    frame_shift = torch.randint(
                        -(self.temporal_stride // 2), self.temporal_stride // 2, (1,)
                    ).item()
                else:
                    frame_shift = torch.randint(
                        -(self.temporal_context * self.temporal_stride - 1),
                        self.temporal_context * self.temporal_stride,
                        (1,),
                    ).item()
                frame_tmp = frame + frame_shift
                path = os.path.join(clip, str(frame_tmp).zfill(8) + ".jpg")
            frame = frame_tmp
        img_annotations = self.annotations.get_group(path).iloc[0]

        # get current frame num
        curr_frame_nb = frame

        # read current frame
        # Annotation keys store show names with underscores for spaces
        # (e.g. "A_Play_With_Words"); restore for the filesystem path.
        img_path = os.path.join(
            self.root, "images",
            path.split("/")[0].replace("_", " "),
            *path.split("/")[1:],
        )
        image = Image.open(img_path)
        img_w, img_h = image.size
        if self.split == "test" and self.aspect:  # maintain aspect ratio
            dummy_sample = {}
            dummy_sample["image"] = image
            dummy_sample["heads"] = []
            dummy_sample = Resize(
                img_size=self.image_size[1], head_size=self.head_size
            )(dummy_sample)
            self.image_size = dummy_sample["image"].size

        # get frame nums around current frame
        frame_nbs = np.arange(
            curr_frame_nb - (self.temporal_stride * self.temporal_context),
            curr_frame_nb + (self.temporal_stride * self.temporal_context) + 1,
            self.temporal_stride,
        )

        # load person ids and head bboxes for current frame
        person_ids = img_annotations["person_ids"]
        inout = img_annotations["inout"]

        # shuffle person ids during training
        if self.split == "train":
            shuffle_idx = get_shuffle_idx(inout)
            person_ids = person_ids[shuffle_idx]

        # keep up to num_people ids
        num_heads = len(person_ids)
        num_keep = num_heads
        if self.num_people != "all":
            batch_num_heads = self.num_people
            if num_heads > 1:
                num_keep = np.random.randint(2, min(num_heads, self.num_people) + 1)
        else:
            batch_num_heads = num_heads
        person_ids = person_ids[-num_keep:]
        num_pairs = batch_num_heads * (batch_num_heads + 1)

        # randomly choose to apply the horizontal flip augmentation
        self.horizontal_flip = False
        if self.split == "train" and torch.rand(1) <= 0.5:
            self.horizontal_flip = RandomHorizontalFlip(p=1)

        # define temporal sample
        t_sample = {
            "image": [],
            "heads": [],
            "head_centers": [],
            "head_masks": [],
            "head_bboxes": [],
            "inout": [],
            "gaze_pts": [],
            "gaze_vecs": [],
            "gaze_heatmaps": [],
            "lah_labels": [],
            "laeo_labels": [],
            "coatt_labels": [],
            "speaking": [],
            "is_child": [],
            "num_valid_people": [],
            "img_size": [],
            "path": [],
            "dataset": "videoattentiontarget",
        }
        t_sample["pids"] = torch.cat(
            [
                torch.zeros((batch_num_heads + 1 - len(person_ids),)) - 1,
                torch.from_numpy(person_ids),
            ]
        )
        for frame_nb in frame_nbs:
            # check if frame exists
            path = os.path.join(clip, str(frame_nb).zfill(8) + ".jpg")
            if path not in self.annotations.groups.keys():
                t_sample["image"].append(
                    torch.zeros(
                        (3, self.image_size[1], self.image_size[0]), dtype=torch.float32
                    )
                )
                t_sample["heads"].append(
                    torch.zeros((batch_num_heads + 1, 3, 224, 224), dtype=torch.float32)
                )
                t_sample["head_centers"].append(
                    torch.zeros((batch_num_heads + 1, 2), dtype=torch.float32)
                )
                t_sample["head_masks"].append(
                    torch.zeros(
                        (
                            batch_num_heads + 1,
                            1,
                            self.image_size[1],
                            self.image_size[0],
                        ),
                        dtype=torch.float32,
                    )
                )
                t_sample["head_bboxes"].append(
                    torch.zeros((batch_num_heads + 1, 4), dtype=torch.float32)
                )
                t_sample["gaze_pts"].append(
                    torch.zeros((batch_num_heads + 1, 2), dtype=torch.float32) - 1
                )
                t_sample["gaze_vecs"].append(
                    torch.zeros((batch_num_heads + 1, 2), dtype=torch.float32)
                )
                t_sample["gaze_heatmaps"].append(
                    torch.zeros(
                        (batch_num_heads + 1, self.heatmap_size, self.heatmap_size),
                        dtype=torch.float32,
                    )
                )
                t_sample["inout"].append(
                    torch.zeros((batch_num_heads + 1), dtype=torch.float32) - 1
                )
                t_sample["lah_labels"].append(
                    torch.zeros((num_pairs), dtype=torch.float32) - 1
                )
                t_sample["laeo_labels"].append(
                    torch.zeros((num_pairs), dtype=torch.float32) - 1
                )
                t_sample["coatt_labels"].append(
                    torch.zeros((num_pairs), dtype=torch.float32) - 1
                )
                t_sample["speaking"].append(
                    torch.zeros((batch_num_heads + 1), dtype=torch.float32) - 1
                )
                t_sample["is_child"].append(
                    torch.zeros((batch_num_heads + 1), dtype=torch.float32) - 1
                )
                t_sample["num_valid_people"].append(torch.zeros(1, dtype=torch.long))
                t_sample["img_size"].append(torch.zeros(2, dtype=torch.long))
                t_sample["path"].append("")
            else:
                ###########################################
                # Get annotations
                ###########################################
                # Load image
                path = os.path.join(clip, str(frame_nb).zfill(8) + ".jpg")
                img_path = os.path.join(
                    self.root, "images",
                    path.split("/")[0].replace("_", " "),
                    *path.split("/")[1:],
                )
                image = Image.open(img_path)
                img_w, img_h = image.size

                # Load annotations for selected person ids
                img_annotations = self.annotations.get_group(path).iloc[0]
                pids_frame = img_annotations["person_ids"]
                pids_selected = []
                head_bboxes = []
                gaze_pts = []
                inout = []
                for pi, pid in enumerate(person_ids):
                    pid_idx = np.where(pids_frame == pid)[0]
                    if len(pid_idx) == 0:
                        pids_selected.append(-1)
                        head_bboxes.append(torch.zeros(4, dtype=torch.float32))
                        gaze_pts.append(torch.zeros(2, dtype=torch.float32) - 1)
                        inout.append(-1)
                    else:
                        pids_selected.append(pid)
                        head_bbox = img_annotations["head_bboxes"][pid_idx]
                        head_bbox = torch.from_numpy(
                            head_bbox.astype(np.float32)
                        ).squeeze()
                        head_bboxes.append(head_bbox)
                        gaze_pt = img_annotations["gaze_points"][pid_idx]
                        gaze_pt = torch.from_numpy(gaze_pt.astype(np.float32)).squeeze()
                        gaze_pts.append(gaze_pt)
                        io = img_annotations["inout"][pid_idx].item()
                        inout.append(io)

                # stack annotations
                pids_selected = torch.tensor(pids_selected, dtype=torch.long)
                head_bboxes = torch.stack(head_bboxes)
                head_bboxes = head_bboxes * torch.tensor(
                    [img_w, img_h, img_w, img_h], dtype=torch.float
                )
                gaze_pts = torch.stack(gaze_pts)
                inout = torch.tensor(inout, dtype=torch.float)

                # jitter head bboxes
                if self.split == "train":
                    head_bboxes = self.jitter_bbox(head_bboxes, img_w, img_h)

                # Square head bboxes (can have negative values)
                head_bboxes = square_bbox(head_bboxes, img_w, img_h)

                # Extract Heads
                heads = []
                for head_bbox in head_bboxes:
                    heads.append(image.crop(head_bbox.int().tolist()))
                num_valid_heads = len(heads)
                #                 num_missing_heads = max(self.num_people - num_valid_heads, 0) if self.num_people != "all" else 0
                num_missing_heads = (
                    max(self.num_people + 1 - num_valid_heads, 1)
                    if self.num_people != "all"
                    else 1
                )  # pad at least one person

                # Normalize Head Bboxes
                head_bboxes /= torch.tensor([img_w, img_h, img_w, img_h], dtype=float)

                # Build Sample
                sample = {
                    "image": image,
                    "heads": heads,
                    "head_bboxes": head_bboxes,
                    "inout": inout,
                    "gaze_pts": gaze_pts,
                    "num_valid_people": torch.tensor([num_valid_heads]),
                    "img_size": torch.tensor((img_w, img_h), dtype=torch.long),
                    "path": path,
                }

                # Transform
                if self.transform:
                    sample = self.transform(sample)
                if self.horizontal_flip:
                    sample = self.horizontal_flip(sample)

                head_bboxes = sample["head_bboxes"]
                gaze_pts = sample["gaze_pts"]
                heads = sample["heads"]

                # Pad missing people (ie. heads, head_bboxes, gaze_pts)
                if num_missing_heads > 0:
                    pids_selected = torch.cat(
                        [
                            torch.zeros((num_missing_heads,), dtype=torch.long) - 1,
                            pids_selected,
                        ]
                    )
                    head_bboxes = torch.cat(
                        [
                            torch.zeros((num_missing_heads, 4), dtype=torch.float32),
                            head_bboxes,
                        ]
                    )
                    heads = torch.cat(
                        [
                            torch.zeros(
                                (num_missing_heads, 3, 224, 224), dtype=torch.float32
                            ),
                            heads,
                        ]
                    )
                    gaze_pts = torch.cat(
                        [
                            torch.zeros((num_missing_heads, 2), dtype=torch.float32)
                            - 1,
                            gaze_pts,
                        ]
                    )
                    inout = torch.cat(
                        [
                            torch.zeros((num_missing_heads,), dtype=torch.float32) - 1,
                            inout,
                        ]
                    )

                # Get social gaze labels
                pairs_all = img_annotations["pairs"]
                lah_labels_all = img_annotations["lah_pairs"]
                laeo_labels_all = img_annotations["laeo_pairs"]
                coatt_labels_all = img_annotations["coatt_pairs"]
                lah_labels = get_lah_labels(pids_selected, pairs_all, lah_labels_all)
                laeo_labels = get_lah_labels(pids_selected, pairs_all, laeo_labels_all)
                coatt_labels = get_lah_labels(
                    pids_selected, pairs_all, coatt_labels_all
                )

                # compute gaze vectors
                head_centers = torch.hstack(
                    [
                        (head_bboxes[:, [0]] + head_bboxes[:, [2]]) / 2,
                        (head_bboxes[:, [1]] + head_bboxes[:, [3]]) / 2,
                    ]
                )
                sample["head_centers"] = head_centers
                sample["gaze_vecs"] = F.normalize(gaze_pts - head_centers, p=2, dim=-1)

                # compute head masks
                _, img_h, img_w = sample["image"].shape
                sample["head_masks"] = generate_mask(head_bboxes, img_w, img_h)

                # generate gaze heatmaps
                sample["gaze_heatmaps"] = generate_gaze_heatmap(
                    gaze_pts, sigma=self.heatmap_sigma, size=self.heatmap_size
                )

                is_child = torch.zeros(len(heads), dtype=torch.float) - 1
                speaking_scores = torch.zeros(len(heads), dtype=torch.float) - 1

                sample["lah_labels"] = lah_labels
                sample["laeo_labels"] = laeo_labels
                sample["coatt_labels"] = coatt_labels
                sample["head_bboxes"] = head_bboxes
                sample["gaze_pts"] = gaze_pts
                sample["heads"] = heads
                sample["inout"] = inout
                sample["speaking"] = speaking_scores

                # Append current frame annotations to temporal sample
                t_sample["image"].append(sample["image"])
                t_sample["heads"].append(sample["heads"])
                t_sample["head_centers"].append(sample["head_centers"])
                t_sample["head_masks"].append(sample["head_masks"])
                t_sample["head_bboxes"].append(sample["head_bboxes"])
                t_sample["gaze_pts"].append(sample["gaze_pts"])
                t_sample["gaze_vecs"].append(sample["gaze_vecs"])
                t_sample["gaze_heatmaps"].append(sample["gaze_heatmaps"])
                t_sample["inout"].append(sample["inout"])
                t_sample["lah_labels"].append(sample["lah_labels"])
                t_sample["laeo_labels"].append(sample["laeo_labels"])
                t_sample["coatt_labels"].append(sample["coatt_labels"])
                t_sample["speaking"].append(sample["speaking"])
                t_sample["is_child"].append(is_child)
                t_sample["num_valid_people"].append(sample["num_valid_people"])
                t_sample["img_size"].append(sample["img_size"])
                t_sample["path"].append(path)

        for key, item in t_sample.items():
            if key not in ["dataset", "path", "pids"]:
                t_sample[key] = torch.stack(t_sample[key], axis=0).squeeze()
                if self.temporal_context == 0:
                    t_sample[key] = t_sample[key].unsqueeze(0)
        return t_sample

    def __len__(self):
        return len(self.paths)
