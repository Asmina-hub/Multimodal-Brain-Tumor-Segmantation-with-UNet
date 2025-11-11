import nibabel as nib
from dotenv import load_dotenv  
load_dotenv()  
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
from sklearn.preprocessing import MinMaxScaler
import random
import glob
from tensorflow.keras.utils import to_categorical

data = []
cwd = os.getcwd()

def load_nii_files( t2_list, t1ce_list, flair_list, mask_list,training_ratio=0.8,training=True):
    mask = []
    train_images_dir = 'dataset/BraTS2020_TrainingData/input_data_3channels/images_train'
    train_masks_dir = 'dataset/BraTS2020_TrainingData/input_data_3channels/masks_train'
    val_images_dir = 'dataset/BraTS2020_TrainingData/input_data_3channels/images_val'
    val_masks_dir = 'dataset/BraTS2020_TrainingData/input_data_3channels/mask_val'
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_masks_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_masks_dir, exist_ok=True)

    for i in range(0,int(training_ratio * len(t1ce_list))):
        slices = []
        temp_image_t2=nib.load(t2_list[i]).get_fdata()
        temp_image_t2 = minmaxscaler_image(temp_image_t2)
        temp_image_t1ce=nib.load(t1ce_list[i]).get_fdata()
        temp_image_t1ce =minmaxscaler_image(temp_image_t1ce)
        temp_image_flair=nib.load(flair_list[i]).get_fdata()
        temp_image_flair =minmaxscaler_image(temp_image_flair)
        temp_mask=nib.load(mask_list[i]).get_fdata()
        temp_mask = change_mask_values(temp_mask)
        slices.append(temp_image_flair)
        slices.append(temp_image_t1ce)
        slices.append(temp_image_t2)
        combine_x, combine_mask = combine_slices(slices,temp_mask)
        val, counts = np.unique(combine_mask, return_counts=True)
    
        if (1 - (counts[0]/counts.sum())) > 0.01:  #At least 1% useful volume with labels that are not 0
            print("Save Me")
            temp_mask= to_categorical(combine_mask, num_classes=4)
            np.save(os.path.join(train_images_dir, f"image_{i}.npy"), combine_x)
            np.save(os.path.join(train_masks_dir, f"mask_{i}.npy"), combine_mask)
            
        else:
            print("I am useless")  


    for i in range(int(training_ratio*len(t1ce_list)), len(t1ce_list)):
        slices = []
        temp_image_t2=nib.load(t2_list[i]).get_fdata()
        temp_image_t2 = minmaxscaler_image(temp_image_t2)
        temp_image_t1ce=nib.load(t1ce_list[i]).get_fdata()
        temp_image_t1ce =minmaxscaler_image(temp_image_t1ce)
        temp_image_flair=nib.load(flair_list[i]).get_fdata()
        temp_image_flair =minmaxscaler_image(temp_image_flair)
        temp_mask=nib.load(mask_list[i]).get_fdata()
        temp_mask = change_mask_values(temp_mask)
        slices.append(temp_image_flair)
        slices.append(temp_image_t1ce)
        slices.append(temp_image_t2)
        combine_x, combine_mask = combine_slices(slices,temp_mask)
        val, counts = np.unique(combine_mask, return_counts=True)
    
        if (1 - (counts[0]/counts.sum())) > 0.01:  #At least 1% useful volume with labels that are not 0
            print("Save Me")
            temp_mask= to_categorical(combine_mask, num_classes=4)
            np.save(os.path.join(val_images_dir, f"image_{i}.npy"), combine_x)
            np.save(os.path.join(val_masks_dir, f"mask_{i}.npy"), combine_mask)
            
        else:
            print("I am useless")

         


def minmaxscaler_image(image):
    scalar = MinMaxScaler()
    reshape_img = image.reshape(-1, 1)
    scalar_imag = scalar.fit_transform(reshape_img).reshape(image.shape)
    return scalar_imag


def change_mask_values(mask):
    test_mask = mask.astype(np.uint8)
    test_mask[test_mask==4] = 3 
    return test_mask

def combine_slices(slices,mask):
    combined_x = np.stack(slices, axis=3) 
    combine_x = combined_x[56:184, 56:184, 13:141]
    combine_mask = mask[56:184, 56:184, 13:141]
    return combine_x, combine_mask

def plot_combined_slices(combined_x, test_mask):
    n_slice=random.randint(0, test_mask.shape[2])
    plt.figure(figsize=(12, 8))
    plt.subplot(221)
    plt.imshow(combined_x[:,:,n_slice, 0], cmap='gray')
    plt.title('Image flair')
    plt.subplot(222)
    plt.imshow(combined_x[:,:,n_slice, 1], cmap='gray')
    plt.title('Image t1ce')
    plt.subplot(223)
    plt.imshow(combined_x[:,:,n_slice, 2], cmap='gray')
    plt.title('Image t2')
    plt.subplot(224)
    plt.imshow(test_mask[:,:,n_slice])
    plt.title('Mask')
    plt.show()

def get_data():
    t2_list = sorted(glob.glob('dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*t2.nii'))
    t1ce_list = sorted(glob.glob('dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*t1ce.nii'))
    flair_list = sorted(glob.glob('dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*flair.nii'))
    mask_list = sorted(glob.glob('dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*seg.nii'))
    return t2_list, t1ce_list, flair_list, mask_list




    
def main():

    datapath = 'dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_074'  # Replace with your NIfTI folder path
    t2_list, t1ce_list, flair_list, mask_list= get_data()
    load_nii_files(t2_list, t1ce_list, flair_list, mask_list)
   

if __name__ == "__main__":
    main()