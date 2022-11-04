import datetime, os
import cv2
import numpy as np
import random 

from collections import defaultdict



class EDA:

    def __init__(self, cats, dir_path):

        self.categories = cats
        self.dirname = dir_path
        self.total_images = 0
        self.image_data = self.load_full_data_set()
        self.image_dim_dict = self.create_dim_dict()


    # Load full image set and append class labels
    def load_full_data_set(self):

        data = []
        for category in self.categories:
            path = os.path.join(self.dirname, category)
            num_imgs = len(os.listdir(path))
            self.total_images += num_imgs
            print('There were {} images loaded in from the {} data set' .format(num_imgs, category))
            class_num = self.categories.index(category)
            for img in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    #img_resize = cv2.resize(img_array, (resize, resize))
                    data.append([img_array, class_num])
                except Exception as e:
                    pass

        return data 


    def create_dim_dict(self):

        # Create a dictionary with a count of each image dimension in the data set
        np_image_data = np.array(self.image_data, dtype=object)
        np_images = np_image_data[:,0]
        dim_dict = defaultdict(int)
        for np_img in np_images:
            dim_dict[np_img.shape] += 1

        temp_dict = {}
        # Get the area of each image dimension in the data set
        img_dim_list = list(dim_dict.keys())
        for item in img_dim_list:
            size = item[0] * item[1]
            temp_dict[item] = [size]

        # Merge the two dictionaries together
        for img in img_dim_list:
            temp_dict[img].append(dim_dict[img])

        return temp_dict


    def get_dim_stats(self):

        # Find the dimension with the highest number of occurrences
        max_val = max(self.image_dim_dict.values(), key=lambda e: e[1])

        key_list = list(self.image_dim_dict.keys())
        val_list = list(self.image_dim_dict.values())

        position = val_list.index(max_val)
        result = key_list[position]

        return result, max_val[1]


    def convert_full_data_set(self, dim, channels):

        # Load full image set and append class labels
        data = []
        for img in self.image_data:
            try:
                image = img[0]
                class_num = img[1]
                img_resize = cv2.resize(image, (dim, dim))
                data.append([img_resize, class_num])
            except Exception as e:
                pass
        
        # Shuffle the data
        random.shuffle(data)
        for sample in data[:10]:
            print('Class Label Num: {} ' .format(sample[1]))


        X = []
        y = []
        # Reshape the data so that it has the correct number of channels for model input
        for features, label in data:
            X.append(features)
            y.append(label)

        X = np.array(X)
        X = X.reshape(len(data), dim, dim, channels)

        return X, y



    def get_data(self):
        return self.image_data

    def get_dim_dict(self):
        return self.image_dim_dict

    def get_total_images(self):
        return self.total_images