import os
import sys
import torch
import imagehash
import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path
from random import sample
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score)

module_path = Path(__file__).parents[1]
sys.path.append(str(module_path))

from ML_Pipeline.Config import *


def remove_duplicate_images(self):

    for image_set in IMAGE_FOLDERS:

        input_folder = self.dirname + image_set
        filenames = os.listdir(input_folder)

        pixel_data = \
            [imagehash.average_hash(Image.open(os.path.join(input_folder,
                                                            file)),
                                    self.hash_size)
                                    for file in filenames]
        retain_image_dict = {image_data: image
                             for (image_data, image)
                             in zip(pixel_data, filenames)}

        files_to_keep = set(retain_image_dict.values())
        all_filenames = set(filenames)
        files_to_remove = all_filenames - files_to_keep

        if(files_to_remove):
            for file in files_to_remove:
                os.remove(os.path.join(input_folder, file))
                print(f"{file} successfully removed!")

        self.initial_image_counts[image_set] = len(files_to_keep)


def augment_image(self, image, image_set):

    path = self.dirname + image_set + '/'
    location = path + image
    imageObject = Image.open(location)
    flippedImage = imageObject.transpose(Image.FLIP_LEFT_RIGHT)
    new_name = AUG_DATA_PREFIX + image
    output = path + new_name
    flippedImage.save(output)


def augment_input_images(self, image_set, images_to_augment):

    list(map(lambda image: self.augment_image(image, image_set),
             images_to_augment))


def balance_input_image_sets(self, imbalanced_dataset_list, max_count):

    # Randomly find images in the dataset to augment for balancing
    for image_set in imbalanced_dataset_list:

        rand_image_dict = \
            {image_set: sample(os.listdir(self.dirname + image_set + '/'),
             max_count - self.initial_image_counts[image_set])}
        
    list(map(lambda item: self.augment_input_images(item[0], item[1]),
             rand_image_dict.items()))


def handle_imbalanced_input_data(self):

    # check to see if any of the input folders have are disproportionately imbalanced
    largest_group = max(self.initial_image_counts, key=self.initial_image_counts.get)
    top_count = self.initial_image_counts[largest_group]
    highly_imbalanced_image_sets = \
        [k for k,v in self.initial_image_counts.items()
         if (top_count-v) >= AUGMENT_DIFF_THRESH]
    
    if(highly_imbalanced_image_sets):
        self.balance_input_image_sets(highly_imbalanced_image_sets, top_count)


def load_processed_dataset(self, type='conv'):

    path = f"{type}_features_path"
    labels = f"{type}_labels_path"

    self.__setattr__(f"{type}_X", torch.load(self.__getattribute__(path)))
    self.__setattr__(f"{type}_y", torch.load(self.__getattribute__(labels)))
    

def load_processed_datasets(self):

    for type in ['conv', 'cap']:

        self.__setattr__(f"{type}_features_path", 
                         os.path.join(module_path,
                                      f"Output/Processed_Data/{type}_full_data_features.pt"))
        
        self.__setattr__(f"{type}_labels_path", 
                         os.path.join(module_path,
                                      f"Output/Processed_Data/{type}_full_data_labels.pt"))
        
        self.load_processed_dataset(type)


def create_weight_dict(self):
    
    # In the event the data was not balanced through augmentation 
    weight_classes = \
        class_weight.compute_class_weight(class_weight='balanced',
                                          classes=np.unique(self.label_data),
                                          y=self.label_data)

    self.class_weights = \
        {i: weight_classes[i] for i in range(len(weight_classes))}
    

def split_dataset(self, type='conv'):

    feature_data = self.__getattribute__(f"{type}_X")
    labels_data = self.__getattribute__(f"{type}_y")

    X_train, X_test, y_train, y_test = \
        train_test_split(feature_data,
                         labels_data,
                         test_size=0.2,
                         random_state=25)

    self.__setattr__(f"{type}_X_train", X_train)
    self.__setattr__(f"{type}_X_test", X_test)
    self.__setattr__(f"{type}_y_train", y_train)
    self.__setattr__(f"{type}_y_test", y_test)

    if(type == 'cap'):

        X_test, X_val, y_test, y_val = \
            train_test_split(self.cap_X_test,
                             self.cap_y_test,
                             test_size = 0.127,
                             random_state = 25)
        
        self.__setattr__(f"{type}_X_val", X_val)
        self.__setattr__(f"{type}_X_test", X_test)
        self.__setattr__(f"{type}_y_val", y_val)
        self.__setattr__(f"{type}_y_test", y_test)

        print(f"Length of {type}_train is {len(X_train)}")
        print(f"Length of {type}_test is {len(X_test)}")
        print(f"Length of {type}_val is {len(self.cap_X_val)}")

    else:

        print(f"Length of {type}_train is {len(X_train)}")
        print(f"Length of {type}_test is {len(X_test)}")


def split_datasets(self):

    self.split_dataset('conv')
    self.split_dataset('cap')


def scale_convolutional_dataset(self):

    X_train_data = self.conv_X_train.astype('float32') 
    X_test_data = self.conv_X_test.astype('float32')

    sc = StandardScaler() # avoiding data leakage on the test set

    len_train_samples, image_height, image_len, pixel_channels = X_train_data.shape

    # Put each pixel into it's own separate list for scaling algorithm
    X_train_flattened = \
        self.conv_X_train.reshape(len_train_samples * image_height * image_len,
                                  pixel_channels)
    
    X_train_flattened_scaled = sc.fit_transform(X_train_flattened)
    self.conv_scaler_map = {'mean':sc.mean_, 'std':np.sqrt(sc.var_)}

    # Reshape the data back to original format for model processing
    self.conv_X_train_scaled = \
        X_train_flattened_scaled.reshape(len_train_samples,
                                         image_height,
                                         image_len,
                                         pixel_channels)
    
    len_test_samples = X_test_data.shape[0]

    X_test_flattened = \
        self.conv_X_test.reshape(len_test_samples * image_height * image_len,
                                 pixel_channels)
    
    X_test_flattened_scaled = sc.transform(X_test_flattened)

    self.conv_X_test_scaled = \
        X_test_flattened_scaled.reshape(len_test_samples,
                                         image_height,
                                         image_len,
                                         pixel_channels)


def scale_capsule_dataset(self):

    self.cap_X_train_scaled = self.cap_X_train.astype('float32') / 255
    self.cap_X_test_scaled = self.cap_X_test.astype('float32') / 255
    self.cap_X_val_scaled = self.cap_X_val.astype('float32') / 255


def scale_features(self):

    self.scale_convolutional_dataset()
    self.scale_capsule_dataset()


def encode_labels(self):

    self.conv_y_train_OHE = tf.keras.utils.to_categorical(self.conv_y_train)
    self.conv_y_test_OHE = tf.keras.utils.to_categorical(self.conv_y_test)

    self.cap_y_train = np.asarray(self.cap_y_train, dtype=np.int32)
    self.cap_y_test = np.asarray(self.cap_y_test, dtype=np.int32)
    self.cap_y_val = np.asarray(self.cap_y_val, dtype=np.int32)


def print_classification_scores(self, type='conv'):

    test_prediction_list = \
        self.__getattribute__(f"{type}_test_prediction_list")
    test_label_list = self.__getattribute__(f"{type}_y_test")

    accuracy = \
        accuracy_score(test_label_list, test_prediction_list)
    print(f'{type} accuracy: {accuracy:.4f} %')

    precision = \
        precision_score(test_label_list,
                        test_prediction_list,
                        average='weighted')
    print(f'{type} precision: {precision:.4f} %' )

    recall = \
        recall_score(test_label_list,
                     test_prediction_list,
                     average='weighted')
    print(f'{type} recall: {recall:.4f} %')

    f1 = \
        f1_score(test_label_list,
                 test_prediction_list,
                 average='weighted')
    print(f'{type} f1 score: {f1:.4f} %')
