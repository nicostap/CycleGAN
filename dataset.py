from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class BatikImageDataset(Dataset):
    def __init__(self, root_image, root_batik, transform=None):
        self.root_image = root_image
        self.root_batik = root_batik
        self.transform = transform

        self.image_images = os.listdir(root_image)
        self.batik_images = os.listdir(root_batik)
        self.length_dataset = max(len(self.image_images), len(self.batik_images))
        self.image_len = len(self.image_images)
        self.batik_len = len(self.batik_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        image_img = self.image_images[index % self.image_len]
        batik_img = self.batik_images[index % self.batik_len]

        image_path = os.path.join(self.root_image, image_img)
        batik_path = os.path.join(self.root_batik, batik_img)

        image_img = np.array(Image.open(image_path).convert("RGB"))
        batik_img = np.array(Image.open(batik_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=image_img, image0=batik_img)
            image_img = augmentations["image"]
            batik_img = augmentations["image0"]

        return image_img, batik_img





