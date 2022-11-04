from PIL import Image
import imagehash
import os
import numpy as np

from random import sample
from keras.preprocessing.image import ImageDataGenerator



CATEGORIES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]


    
class Images:

    def __init__(self, dirname, hash_size = 250):

        self.dirname = dirname
        self.hash_size = hash_size


        
    def find_duplicates(self, category):
        """
        Find and Delete Duplicate Images 
        """
        path = self.dirname + category                    
        fnames = os.listdir(path)
        hashes = {}
        duplicates = []
        for image in fnames:
            with Image.open(os.path.join(path,image)) as img:
                temp_hash = imagehash.average_hash(img, self.hash_size)
                if temp_hash in hashes:
                    print("Duplicate {} \nfound for Image {}!\n".format(image,hashes[temp_hash]))
                    duplicates.append(image)
                else:
                    hashes[temp_hash] = image
                   
        if len(duplicates) != 0:
            a = input("Do you want to delete these {} Images? Press Y or N:  ".format(len(duplicates)))
            space_saved = 0
            if(a.strip().lower() == "y"):
                for duplicate in duplicates:
                    space_saved += os.path.getsize(os.path.join(path,duplicate))
                    
                    os.remove(os.path.join(path,duplicate))
                    print("{} Deleted Succesfully!".format(duplicate))
    
                print("\n\nYou saved {} mb of Space!".format(round(space_saved/1000000),2))
            else:
                print("Thank you for Using Duplicate Remover")
        else:
            print(f"No duplicates found in the {category} folder. ")
            
        
               
    def find_similar(self, location, similarity=80):

        fnames = os.listdir(self.dirname)
        threshold = 1 - similarity/100
        diff_limit = int(threshold*(self.hash_size**2))
        
        with Image.open(location) as img:
            hash1 = imagehash.average_hash(img, self.hash_size).hash
        
        print("Finding Similar Images to {} Now!\n".format(location))
        for image in fnames:
            with Image.open(os.path.join(self.dirname,image)) as img:
                hash2 = imagehash.average_hash(img, self.hash_size).hash
                
                if np.count_nonzero(hash1 != hash2) <= diff_limit:
                    print("{} image found {}% similar to {}".format(image,similarity,location))



    # Image Augmentation to create new images to balance the dataset     
    def balancing_augmentation(self, d_counts, prefix):
    
        for category in d_counts.keys():
        
            num_new = d_counts[category]
            path = self.dirname + category + '/'                  
            fnames = os.listdir(path)
            rand_images = sample(fnames, num_new)
            for image in rand_images:

                location = path + image 
                imageObject = Image.open(location)
                flippedImage = imageObject.transpose(Image.FLIP_LEFT_RIGHT)
                new_name = prefix + '_' + image
                output = path + new_name
                flippedImage = flippedImage.save(output)


    # Image Augmentation to create new images to proportionally supplement the data set  
    def manual_augmentation(self, img_counts, prefix, add_percentage, trans_folder):
    
        for category in img_counts:
        
            count = img_counts[category]
            add_count = int(count * add_percentage)
            path = self.dirname + category + '/'                  
            fnames = os.listdir(path)
            rand_images = sample(fnames, add_count)
            for image in rand_images:

                location = path + image 
                imageObject = Image.open(location)
                flippedImage = imageObject.transpose(Image.FLIP_TOP_BOTTOM)
                new_name = prefix + '_' + image
                output = trans_folder + category + '/' + new_name
                flippedImage = flippedImage.save(output)


    def rename_images(self):
    
        for category in CATEGORIES:

            path = self.dirname + category                    
            fnames = os.listdir(path)
            count = 0
            for fname in fnames:

                dst = f"{category}_{str(count)}.jpg"
                src =f"{self.dirname}{category}/{fname}" 
                dst =f"{self.dirname}{category}/{dst}"
         
                os.rename(src, dst) 
                count += 1