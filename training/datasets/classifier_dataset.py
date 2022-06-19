import math
import os
import random
import sys
import traceback

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import dlib
import matplotlib.pyplot as plt
from training.datasets.validation_set import PUBLIC_SET

class DeepFakeClassifierDataset(Dataset):

    def __init__(self,
                 data_path="/mnt/sota/datasets/deepfake",
                 fold=0,
                 label_smoothing=0.01,
                 padding_part=3,
                 hardcore=True,
                 crops_dir="crops",
                 folds_csv="folds.csv",
                 normalize={"mean": [0.485, 0.456, 0.406],
                            "std": [0.229, 0.224, 0.225]},
                 rotation=False,
                 mode="train",
                 reduce_val=True,
                 oversample_real=False,
                 transforms=None
                 ):
        super().__init__()
        self.data_root = data_path
        self.fold = fold
        self.folds_csv = folds_csv
        self.mode = mode
        self.rotation = rotation
        self.padding_part = padding_part
        self.hardcore = hardcore
        self.crops_dir = crops_dir
        self.label_smoothing = label_smoothing
        self.normalize = normalize
        self.transforms = transforms
        self.df = pd.read_csv(self.folds_csv)
        self.oversample_real = oversample_real
        self.reduce_val = reduce_val

    def __getitem__(self, index: int):

        while True:
            img_path, label, fold = self.data[index]
            try:
                if self.mode == "train":
                    label = np.clip(label, self.label_smoothing, 1 - self.label_smoothing)
                #img_path = os.path.join(self.data_root, self.crops_dir, video, img_file)
                image = cv2.imread(img_path, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                return {"image": image,
                        "labels": np.array((label,))} # "origin_image": origin_image, "mask":part_mask,
            except Exception as e:
                traceback.print_exc(file=sys.stdout)
                print("Broken image", os.path.join(self.data_root, self.crops_dir, video, img_file))
                index = random.randint(0, len(self.data) - 1)

    def reset(self, epoch, seed):
        self.data = self._prepare_data(epoch, seed)

    def __len__(self) -> int:
        return len(self.data)

    def _prepare_data(self, epoch, seed):
        df = self.df
        if self.mode == "train":
            rows = df[df["fold"] != self.fold]
        else:
            rows = df[df["fold"] == self.fold]
        seed = (epoch + 1) * seed
        if self.oversample_real:
            rows = self._oversample(rows, seed)
        if self.mode == "val" and self.reduce_val:
            # every 2nd frame, to speed up validation
            rows = rows[rows["frame"] % 20 == 0]
            # another option is to use public validation set
            #rows = rows[rows["video"].isin(PUBLIC_SET)]

        print(
            "real {} fakes {} mode {}".format(len(rows[rows["label"] == 0]), len(rows[rows["label"] == 1]), self.mode))
        data = rows.values

        np.random.seed(seed)
        np.random.shuffle(data)
        return data
