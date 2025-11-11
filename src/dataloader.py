import torch
import numpy as np
from  torch.utils.data import Dataset, DataLoader
import os

class CustomDataset(Dataset):
    def __init__(self, image_npy_files, mask_npy_files, transform=None):
        # Accept either a directory path or a list of file paths.
        self.image_files = sorted([os.path.join(image_npy_files, f) for f in os.listdir(image_npy_files) if f.endswith('.npy')])
        self.mask_files = sorted([os.path.join(mask_npy_files, f) for f in os.listdir(mask_npy_files) if f.endswith('.npy')])
        
        if len(self.image_files) != len(self.mask_files):
            raise ValueError(f"Number of image files ({len(self.image_files)}) != mask files ({len(self.mask_files)})")

        self.transform = transform
 

    def __len__(self):
        # return number of samples (use the prepared file list)
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # load by filepath from the prepared lists
        image = np.load(self.image_files[idx])
        mask = np.load(self.mask_files[idx])
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()  # for segmentation labels
        
        # Move channel dimension (C) to first position for PyTorch convention
        if image.ndim == 4:
            image = image.permute(3, 0, 1, 2)  # (C, H, W, D)
        if mask.ndim == 4:
            mask = mask.permute(3, 0, 1, 2)    # (classes, H, W, D)
        
        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask  
   

#train_img_dir = "dataset/BraTS2020_TrainingData/input_data_3channels/images"
#train_mask_dir = "dataset/BraTS2020_TrainingData/input_data_3channels/masks"

#train_dataset = CustomDataset(train_img_dir, train_mask_dir)
#train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Check a batch
#for img, mask in train_loader:
 #   print(f"Image shape: {img.shape}, Mask shape: {mask.shape}")
  #  break

#import random
#import matplotlib.pyplot as plt

#img, mask = train_dataset[random.randint(0, len(train_dataset)-1)]
# If mask is one-hot, collapse; otherwise assume it's already label volume
#if mask.ndim == 4:
 #   mask_vis = torch.argmax(mask, dim=0)
#else:
 #   mask_vis = mask

#n_slice = random.randint(0, mask_vis.shape[2]-1)

#plt.figure(figsize=(12, 8))
#plt.subplot(221)
#plt.imshow(img[0, :, :, n_slice], cmap='gray')
#plt.title('Image flair')
#plt.subplot(222)
#plt.imshow(img[1, :, :, n_slice], cmap='gray')
#plt.title('Image t1ce')
#plt.subplot(223)
#plt.imshow(img[2, :, :, n_slice], cmap='gray')
#plt.title('Image t2')
#plt.subplot(224)
#plt.imshow(mask_vis[:, :, n_slice])
#plt.title('Mask')
#plt.show()




