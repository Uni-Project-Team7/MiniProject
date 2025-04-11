import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class DeblurringDataset(Dataset):
    def __init__(self, blurred_dir, sharp_dir):
        self.blurred_dir = blurred_dir
        self.sharp_dir = sharp_dir
        self.transform = A.Compose(
            [
                A.Normalize(mean=0, std=1, max_pixel_value=255.0),
                ToTensorV2(),
            ],
            additional_targets={"image0": "image"}
        )
        self.image_filenames = sorted(os.listdir(blurred_dir))

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


if __name__ == '__main__' :
    dataset = DeblurringDataset(blurred_dir='/teamspace/studios/this_studio/dataset/train_crops/blur_crops', sharp_dir='/teamspace/studios/this_studio/dataset/train_crops/sharp_crops')
    print(f"Number of train image pairs: {len(dataset)}")
    dataset = DeblurringDataset(blurred_dir='/teamspace/studios/this_studio/dataset/val_crops/blur_crops', sharp_dir='/teamspace/studios/this_studio/dataset/val_crops/sharp_crops')
    print(f"Number of val image pairs: {len(dataset)}")