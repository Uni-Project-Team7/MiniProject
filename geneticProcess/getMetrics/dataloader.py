import os
import cv2
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import torch

class DeblurringDatasetVal(Dataset):
    def __init__(self):
        self.blurred_dir = '/home/nas/dataset/val/input/'
        self.sharp_dir = '/home/nas/dataset/val/target/'

        self.transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.5),
                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent=(-0.0625, 0.0625),
                    rotate=(-15, 15),
                    p=0.5,
                    border_mode=cv2.BORDER_REFLECT
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                A.Normalize(mean=(0.0, 0.0, 0.0),
                            std=(1.0, 1.0, 1.0),
                            max_pixel_value=255.0),
                ToTensorV2()
            ],
            additional_targets={"image0": "image"}
        )
        self.image_filenames = sorted(os.listdir(self.blurred_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        blurred_path = os.path.join(self.blurred_dir, img_name)
        sharp_path = os.path.join(self.sharp_dir, img_name)

        blurred = cv2.imread(blurred_path)
        blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
        sharp = cv2.imread(sharp_path)
        sharp = cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=blurred, image0=sharp)
            blurred = augmented['image']
            sharp = augmented['image0']

        return blurred, sharp


class DeblurringDatasetTrain(Dataset):
    def __init__(self):
        self.blurred_dir = '/home/nas/dataset/train/input/'
        self.sharp_dir = '/home/nas/dataset/train/target/'

        self.patch_size = 256
        self.image_width = 1020
        self.image_height = 720

        self.transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.5),
                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent=(-0.0625, 0.0625),
                    rotate=(-15, 15),
                    p=0.5,
                    border_mode=cv2.BORDER_REFLECT
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                A.Normalize(mean=(0.0, 0.0, 0.0),
                            std=(1.0, 1.0, 1.0),
                            max_pixel_value=255.0),
                ToTensorV2()
            ],
            additional_targets={"image0": "image"}
        )

        self.image_filenames = sorted(os.listdir(self.blurred_dir))

    def __len__(self):
        return len(self.image_filenames)

    def get_non_overlapping_coords(self):
        max_x = self.image_width - self.patch_size
        max_y = self.image_height - self.patch_size
        selected = []

        tries = 0
        while len(selected) < 3 and tries < 100:
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            rect = (x, y, x + self.patch_size, y + self.patch_size)

            overlap = False
            for sx, sy, ex, ey in selected:
                if not (rect[2] <= sx or rect[0] >= ex or rect[3] <= sy or rect[1] >= ey):
                    overlap = True
                    break

            if not overlap:
                selected.append(rect)

            tries += 1

        return selected

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        blurred_path = os.path.join(self.blurred_dir, img_name)
        sharp_path = os.path.join(self.sharp_dir, img_name)

        blurred = cv2.imread(blurred_path)
        blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
        sharp = cv2.imread(sharp_path)
        sharp = cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB)

        coords = self.get_non_overlapping_coords()

        patch_pairs = []
        for x1, y1, x2, y2 in coords:
            b_patch = blurred[y1:y2, x1:x2]
            s_patch = sharp[y1:y2, x1:x2]

            if self.transform:
                augmented = self.transform(image=b_patch, image0=s_patch)
                b_patch = augmented['image']
                s_patch = augmented['image0']

            patch_pairs.append((b_patch, s_patch))

        return patch_pairs

def flatten_patch_collate_fn(batch):
    all_blurred = []
    all_sharp = []
    for sample in batch:
        for b_patch, s_patch in sample:
            all_blurred.append(b_patch)
            all_sharp.append(s_patch)
    return torch.stack(all_blurred), torch.stack(all_sharp)


if __name__ == '__main__':
    train = DeblurringDatasetTrain()
    dataloader = DataLoader(train, batch_size=2, shuffle=True)
    print(f"number of train pairs : {len(dataloader)}")

    test = DeblurringDatasetVal()
    dataloader = DataLoader(test, batch_size=4, shuffle=False)
    print(f"number of test pairs : {len(dataloader)}")

