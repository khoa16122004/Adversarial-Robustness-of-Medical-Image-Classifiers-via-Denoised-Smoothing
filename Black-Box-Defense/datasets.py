from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torchvision.datasets.utils import check_integrity
from typing import *
from torchvision import transforms

import numpy as np
import os
import pickle
import torch

# for MRI Dataset
from pathlib import Path
from scipy.io import loadmat
import copy
import json
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split
from config import *


# set this environment variable to the location of your imagenet directory if you want to read ImageNet data.
# make sure your val directory is preprocessed to look like the train directory, e.g. by running this script
# https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
IMAGENET_LOC_ENV = "IMAGENET_DIR"

# list of all datasets
DATASETS = ["SIPADMEK_Noise", "SIPADMEK", "Brain_Tumor","Brain_Tumor_Noise", "imagenet", "imagenet32", "cifar10", "mnist", "stl10", "restricted_imagenet"]

img_to_tensor = ToTensor()


def get_dataset(dataset: str, split: str) -> Dataset:
    """Return the dataset as a PyTorch Dataset object"""
    if dataset == "imagenet":
        return _imagenet(split)
    
    elif dataset == "Brain_Tumor":
        return BrainTumorDataset(split)
    
    elif dataset == "Brain_Tumor_Noise":
        return BrainTumorDataset_Noise(split,
                                       transform=transforms.Compose([transforms.Resize((384, 384)), 
                                                        transforms.ToTensor()]))
    
    elif dataset == "SIPADMEK":
        if split == "Train":
            return SIPADMEK(img_dir=r"Dataset/SIPADMEK/process",
                            mode=split)
        else:
            return SIPADMEK(img_dir=r"Dataset/SIPADMEK/process",
                            mode=split,
                            transform=transforms.Compose([transforms.Resize((384, 384)), 
                                                        transforms.ToTensor()])
                            )
    elif dataset == "SIPADMEK_Noise":
        return SIPADMEK_Noise(split,
                              transform=transforms.Compose([transforms.Resize((384, 384)), 
                                                        transforms.ToTensor()]))
        
 


def get_num_classes(dataset: str):
    if dataset in ["Brain_Tumor", "Brain_Tumor_Noise"]:
        return 4
    elif dataset in ["SIPADMEK", "SIPADMEK_Noise"]:
        return 3
    


class SIPADMEK(Dataset):
    def extract_data(self, image_dir: str,
                class_name: str,
                output_dir="Dataset/SIPADMEK/process",
                class_map = {
                    "im_Dyskeratotic": 0, # abnormal
                    "im_Koilocytotic": 0, # abnormal
                    "im_Metaplastic": 1, # Benign
                    "im_Parabasal": 2, # normal
                    "im_Superficial-Intermediate": 2, # normal
                        }
                ):
    
    
        os.makedirs(output_dir, exist_ok=True) # check exist
        class_label = class_map[class_name]
        
        label_dir = os.path.join(output_dir, str(class_label))
        os.makedirs(label_dir, exist_ok=True)
        
        count = 0
        for file_name in tqdm(os.listdir(image_dir)):
            if "bmp" in file_name:
                count += 1
                file_path = os.path.join(image_dir, file_name)
                img = Image.open(file_path).convert("RGB")
                base_name = file_name.split(".")[0]
                output_path = os.path.join(label_dir, f"{class_name}{base_name}.png")
                img.save(output_path)
        print(count)


    def split_data(self, img_dir, train_size=0.7, val_size=0.1, test_size=0.2):
        random.seed("22520691")
        train_img, val_img, test_img = [], [], []
        train_label, val_label, test_label = [], [], []
        
        for label_name in os.listdir(img_dir):
            label_folder = os.path.join(img_dir, label_name)
            tmp = []
            tmp_label = []
            
            for file_name in os.listdir(label_folder):
                file_path = os.path.join(label_folder, file_name)
                tmp.append(file_path)
                tmp_label.append(label_name)
            
            combined = list(zip(tmp, tmp_label))
            random.shuffle(combined)
            tmp, tmp_label = zip(*combined)
            
            n_train = int(len(tmp) * train_size)
            n_val = int(len(tmp) * val_size)
            
            train_img += tmp[:n_train]
            val_img += tmp[n_train:n_train + n_val]
            test_img += tmp[n_train + n_val:]
            
            train_label += tmp_label[:n_train]
            val_label += tmp_label[n_train:n_train + n_val]
            test_label += tmp_label[n_train + n_val:]
            
        return train_img, val_img, test_img, train_label, val_label, test_label

    def save_to_txt(self, image_paths, labels, split_name, output_dir):
        txt_file_path = os.path.join(output_dir, f"{split_name}.txt")
        with open(txt_file_path, 'w') as f:
            for img_path, label_name in zip(image_paths, labels):
                f.write(f"{img_path}, {label_name}\n")
    
    
    def __init__(self, img_dir, mode="Train",
                 transform=transforms.Compose([transforms.Resize((384, 384)),
                                               transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2),transforms.RandomPerspective(distortion_scale=0.2),], p=0.3),
                                               transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2),transforms.RandomAffine(degrees=10),], p=0.3),
                                               transforms.RandomVerticalFlip(p=0.3),
                                               transforms.RandomHorizontalFlip(p=0.3),
                                               transforms.ToTensor(),
                                               ])):
        
        self.transform = transform
        
        # train_img, val_img, test_img, train_label, val_label, test_label = self.split_data(img_dir)
        if mode == "Train":
            path_file = os.path.join(img_dir, "train.txt")
        elif mode == "Val":
            path_file = os.path.join(img_dir, "val.txt")
        else:
            path_file = os.path.join(img_dir, "test.txt")
        
        with open(path_file, 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            img_paths = [line.split(", ")[0] for line in lines]
            labels = [line.split(", ")[1] for line in lines]
        self.img_paths, self.labels = img_paths, labels

    def __len__(self) -> int:
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = int(self.labels[index])
        return img, label
class SIPADMEK_Noise(Dataset):    
    def __init__(self, split, # FGSM, DDN, PGD
                 transform=transforms.Compose([transforms.Resize((384, 384)),
                                               transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2),transforms.RandomPerspective(distortion_scale=0.2),], p=0.3),
                                               transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2),transforms.RandomAffine(degrees=10),], p=0.3),
                                               transforms.RandomVerticalFlip(p=0.3),
                                               transforms.RandomHorizontalFlip(p=0.3),
                                               transforms.ToTensor(),
                                               ])):
        
        self.transform = transform
        img_dir = f"Dataset/SIPADMEK/AT_{split}"
        
        img_paths = []
        labels = []
        
        for label_name in os.listdir(img_dir):
            label_dir = os.path.join(img_dir, label_name)
            for file_name in os.listdir(label_dir):
                file_path = os.path.join(label_dir, file_name)
                img_paths.append(file_path)        
                labels.append(label_name)
        
        self.img_paths = img_paths
        self.labels = labels
                
    def __len__(self) -> int:
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = int(self.labels[index])
        # return img, label, os.path.basename(img_path)
        return img, label

class BrainTumorDataset(Dataset):   
    def __init__(self, mode="Train", 
                 transform=transforms.Compose([transforms.Resize((384,384)), 
                                               transforms.ToTensor()])):
        img_paths = []
        labels = []
        
        if mode == "Train" or mode == "Val":
            img_dir = "Dataset/Brain_Tumor/Training"
        elif mode == "Test":
            img_dir = "Dataset/Brain_Tumor/Testing"
        
        for class_name in sorted(os.listdir(img_dir)):
            class_folder = os.path.join(img_dir, class_name)
            for file_name in os.listdir(class_folder):
                file_path = os.path.join(class_folder, file_name)
                img_paths.append(file_path)
                labels.append(label_str2num[class_name])
                
        train_imgs, val_imgs, train_labels, val_labels = train_test_split(
            img_paths, labels, test_size=0.1, random_state=16122004
        )
        
        if mode == "Train":
            self.img_paths = train_imgs
            self.labels = train_labels
        elif mode == "Val":
            self.img_paths = val_imgs
            self.labels = val_labels
        else:
            self.img_paths = img_paths
            self.labels = labels

        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label_ts = self.labels[idx]
        # return img, label_ts, os.path.basename(img_path)
        return img, label_ts
class BrainTumorDataset_Noise(Dataset):
    def __init__(self,
                 split, # ["FGSM", "PGD", "DDN"]
                 transform=transforms.Compose([transforms.Resize((384,384)), 
                                               transforms.ToTensor()])):
        
        
        img_paths = []
        labels = []
        img_dir = f"Dataset/Brain_Tumor/AT_{split}"
        
        for label in os.listdir(img_dir):
            class_folder = os.path.join(img_dir, label)
            for file_name in os.listdir(class_folder):
                file_path = os.path.join(class_folder, file_name)
                img_paths.append(file_path)
                labels.append(label)
        
        
        self.num2label = {0: 'glioma',
                          1: 'meningioma',
                          2: 'notumor',
                          3: 'pituitary',
                          }
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label_ts = int(self.labels[idx])
        # return img, label_ts, os.path.basename(img_path)
        return img, label_ts


