import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class MultiHeadDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt'))

        with open(label_path, "r") as f:
            lines = [line.strip() for line in f.readlines()]
            head_a_label = int(lines[0])
            head_b_labels = [int(x) for x in lines[1].split(",")]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Multi-label one-hot encoding
        head_b_target = torch.zeros(4)
        head_b_target[head_b_labels] = 1.0

        return image, torch.tensor(head_a_label), head_b_target
