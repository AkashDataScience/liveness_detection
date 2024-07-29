import os
import numpy as np
import albumentations
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch.transforms import ToTensorV2

class CustomDataset(Dataset):
    def __init__(self, root, data_df, transform=None):
        self.root = root
        self.data_df=data_df
        self.transform=transform

    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self,index):
        image_path, label = self.data_df.iloc[index]  
        image_path = os.path.join(self.root,"images", image_path) 
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        if self.transform:
            image = self.transform(image=image)['image']
        return image, label
    
def _get_train_transforms():
    train_transforms = albumentations.Compose([
        albumentations.Resize(224, 224, always_apply=True),
        albumentations.ColorJitter(brightness=(0.7, 1), contrast=(0.7, 1), saturation=(0.7, 1)),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.ShiftScaleRotate(),
        albumentations.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.1, 2.0)),
        albumentations.CoarseDropout(max_holes=8, max_height=24, max_width=24, min_holes=1, 
                                     min_height=20, min_width=20),
        albumentations.ToGray(p=0.1),
        albumentations.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225]),
        ToTensorV2()])
    return train_transforms

def _get_test_transforms():
    test_transforms = albumentations.Compose([
        albumentations.Resize(224, 224, always_apply=True),
        albumentations.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225]),
        ToTensorV2()])
    return test_transforms

def get_train_data_loader(root, data_df, **kwargs):
    train_transforms = _get_train_transforms()
    train_data = CustomDataset(root, data_df, train_transforms)
    train_loader = DataLoader(train_data, **kwargs)
    return train_loader

def get_val_test_data_loader(root, df, **kwargs):
    transforms = _get_test_transforms()
    data = CustomDataset(root, df, transforms)
    data_loader = DataLoader(data, **kwargs)
    return data_loader