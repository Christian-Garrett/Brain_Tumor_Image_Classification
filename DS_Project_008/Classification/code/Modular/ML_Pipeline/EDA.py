import os
import cv2
import sys
import torch
import random
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

module_path = Path(__file__).parents[1]
sys.path.append(str(module_path))

from ML_Pipeline.Config import *


def get_image_data(self, img_set, img):

    try:
        img_folder = os.path.join(self.dirname, img_set)
        img_pixel_array = cv2.imread(os.path.join(img_folder, img),
                                     cv2.IMREAD_GRAYSCALE)
        img_shape = img_pixel_array.shape
        self.image_shape_counts[img_shape] += 1
        self.image_shape_areas[img_pixel_array.shape] = \
            [img_shape[0] * img_shape[1]]
        label_encoded_target_var = IMAGE_FOLDERS.index(img_set)
        return [img_pixel_array, label_encoded_target_var]
    
    except Exception as e:
        return None


def load_brain_scans(self):

    # Load image pixel data and append class labels
    self.image_pixel_data = \
        [self.get_image_data(image_set, img)
         for image_set in IMAGE_FOLDERS
         for img in os.listdir(os.path.join(self.dirname, image_set))]
    
    self.total_images = len(self.image_pixel_data)

    self.label_data = np.asarray([i[1] for i in self.image_pixel_data])

    self.target_label_dict = \
        {k:v for k,v in zip(IMAGE_FOLDERS, np.unique(self.label_data))}

    # Dictionary that contains image shape counts, and areas
    self.image_shape_dict = \
        {key: self.image_shape_areas[key] + [self.image_shape_counts[key]]
         for key in self.image_shape_areas}


def get_image_info(self):

    shape_info_tuples = self.image_shape_dict.values()
    top_occuring_shape_info = max(shape_info_tuples, key=lambda e: e[1])
    image_shapes_list = list(self.image_shape_dict.keys())
    shape_info_list = list(shape_info_tuples)
    position = shape_info_list.index(top_occuring_shape_info)
    top_image_shape = image_shapes_list[position]

    print(f'\n The most frequently occuring image shape in the input data \
is {top_image_shape} which shows up \
{round((top_occuring_shape_info[1]/self.total_images)*100,2)}% of the time \
with {top_occuring_shape_info[1]} occurrences.')
    
    # save a test image to an output folder and print out the shape and pixel info
    plt.imshow(self.image_pixel_data[0][0], cmap="gray")
    plt.savefig(os.path.join(module_path, "Output/Data_Exploration/test_img_II_.png"))
    plt.clf()

    print(f'\nTest Image Shape: {self.image_pixel_data[0][0].shape}')
    print(f'Test Image Pixel Min Value: {self.image_pixel_data[0][0].min()}, \
Max Value: {self.image_pixel_data[0][0].max()}\n')
    
    # Save Image Intensity Distribution to the Output Folder
    plt.figure(figsize=(10, 10))
    plt.hist(self.image_pixel_data[0][0], histtype='bar', bins=10)
    plt.title("Histogram of Pixel Intensity")
    plt.savefig(os.path.join(module_path,
                             "Output/Data_Exploration/original_test_img_intensity_hist_.png"))
    plt.clf()


def resize_images(self):  ## todo: add exception handling map function

    self.conv_data = [[cv2.resize(img[0], (self.conv_resize, self.conv_resize)), img[1]]
                       for img in self.image_pixel_data]

    self.cap_data = [[cv2.resize(img[0], (self.cap_resize, self.cap_resize)), img[1]]
                      for img in self.image_pixel_data]    


def shuffle_images(self):
    
    random.shuffle(self.conv_data)
    random.shuffle(self.cap_data)


def reshape_image(self, type='conv'):

    X_data, y_data = f"{type}_X", f"{type}_y"
    dim_size = f"{type}_resize"
    data = self.__getattribute__(f"{type}_data")

    pixel_features, image_labels = zip(*data)
    pixel_features = list(pixel_features)
    self.__setattr__(y_data, list(image_labels))

    pixel_features = np.array(pixel_features)
    pixel_features = \
        pixel_features.reshape(len(data),
                               self.__getattribute__(dim_size),
                               self.__getattribute__(dim_size),
                               self.color_channels)
    
    self.__setattr__(X_data, pixel_features)


def reshape_images(self):

    self.reshape_image('conv')
    self.reshape_image('cap')


def save_processed_image(self, type='conv'):

    feature_name = f"{type}_full_data_features_.pt"
    feature_path = os.path.join(module_path, f"Output/Processed_Data/{feature_name}")
    torch.save(self.__getattribute__(f"{type}_X"), feature_path)

    label_name = f"{type}_full_data_labels_.pt"
    label_path = os.path.join(module_path, f"Output/Processed_Data/{label_name}") 
    torch.save(self.__getattribute__(f"{type}_y"), label_path)


def save_processed_images(self):
    
    self.save_processed_image('conv')
    self.save_processed_image('cap')
