import time
import datetime, os
import cv2
import random 
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
from tensorflow.keras.callbacks import TensorBoard
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from Classification.code.Modular.ML_Pipeline.Tools import CATEGORIES, Images
from Classification.code.Modular.ML_Pipeline.EDA import EDA
from Classification.code.Modular.ML_Pipeline.Convolutional_NN import Convolutional_NN
from Classification.code.Modular.ML_Pipeline.Capsule_NN import Capsule_NN


# Globals

# Convolutional Neural Network Tuning Directory Path
LOG_DIRECTORY = f'Classification/code/Modular/Output/Log_Directory/Convolutional/{int(time.time())}'

# Convolutional Neural Network Checkpointing Path
CONV_CHECKPT_LOG_DIR = 'Classification/code/Modular/Output/Checkpointing/Convolutional'

# Capsule Neural Network Checkpointing Path
CAP_CHECKPT_LOG_DIR = 'Classification/code/Modular/Output/Checkpointing/Capsule/Capsule'

# Image Resizing
CONV_IMG_SIZE = 256
CAP_IMG_SIZE = 56

# Image folder names
CATEGORIES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]




def prepare_model_data():

    # Data Preprocessing - 

    # Image Directory Path
    full_path = 'Classification/code/Modular/Input/Clean_Set_Updated/'

    # Connect to the data set input folder
    image_data = Images(full_path)

    # Check for duplicates in the input folder and remove if necessary
    print("Finding Duplicates Now!\n")
    for category in CATEGORIES:
        sub_folder = full_path + category
        image_data.find_duplicates(category)

    # Get the number of images in each data subset
    Image_Counts = dict()
    print('\n')
    for category in CATEGORIES:
        path = os.path.join(full_path, category)
        Image_Counts[category] = len(os.listdir(path))
        print('There are {} images in the {} data set' .format(len(os.listdir(path)), category))
    print('\n')

    # Create augmented data to balance out the data sets if possible
    largest_group = max(Image_Counts, key=Image_Counts.get)
    max_count = Image_Counts[largest_group]

    # Find number of images needed to balance each of the data subsets
    Diff_Counts = dict()
    for category in CATEGORIES:
        Diff_Counts[category] = Diff_Counts[category] = int(max_count - Image_Counts[category]) 

    image_data.balancing_augmentation(Diff_Counts, 'aug_scan')


    # Directory path for manually added images
    added_imgs = 'Classification/code/Modular/Input/Added/'
    ### image_data.manual_augmentation(Image_Counts, 'flipped_scan', 0.5, added_imgs)


    # Rename images in the data set for consistency
    image_data.rename_images()
 

    # Data Exploration - 

    # load image data
    load_images = EDA(CATEGORIES, full_path)
    image_data = load_images.get_data()

    # get sorted image dimension info: dimensions - [count, area]
    image_dim_dict = load_images.get_dim_dict()
    sorted_dim_dict = sorted(image_dim_dict.items(), key=lambda e: e[0][1], reverse=True)

    # print dimension statistics
    top_dim, top_occur = load_images.get_dim_stats()
    num_images = load_images.get_total_images()
    print(f'\nThe dimension occuring most often', f"({round((top_occur/num_images)*100,2)}%) is {top_dim} with {top_occur} occurrences", '\n')

    # save a test image to an output folder and print out the shape and pixel info
    output = 'Classification/code/Modular/Output/Data_Exploration/test_img_II.png'
    plt.imshow(image_data[0][0], cmap="gray")
    plt.savefig(output)
    plt.clf()

    print('Test Image Shape: {}' .format(image_data[0][0].shape))
    print('Test Image Pixel Min: {}, Max: {}\n' .format(image_data[0][0].min(), image_data[0][0].max()))

    # Save Image Intensity Distribution to the Output Folder
    output = "Classification/code/Modular/Output/Data_Exploration/original_test_img_intensity_hist.png" 
    plt.figure(figsize=(10, 10))
    plt.hist(image_data[0][0], histtype='bar', bins=10)
    plt.title("Histogram of Pixel Intensity")
    plt.savefig(output)
    plt.clf()

    # Save Sample Resized Images to the Output Folder
    output = "Classification/code/Modular/Output/Data_Exploration/resized_test_img_256.png" 
    conv_img_array = cv2.resize(image_data[0][0], (CONV_IMG_SIZE, CONV_IMG_SIZE))
    plt.imshow(conv_img_array, cmap="gray")
    plt.savefig(output)
    plt.clf()

    output = "Classification/code/Modular/Output/Data_Exploration/resized_test_img_56_II.png" 
    cap_img_array = cv2.resize(image_data[0][0], (CAP_IMG_SIZE, CAP_IMG_SIZE))
    plt.imshow(cap_img_array, cmap="gray")
    plt.savefig(output)
    plt.clf()


    # Data Wrangling -

    # Resize, Shuffle and Reshape the Data (convert each pixel array to a single channel for grayscale)
    print('Convolutional Shuffled Data:')    
    conv_full_data_X, conv_full_data_y = load_images.convert_full_data_set(CONV_IMG_SIZE, 1)
    print('\nConvolutional Full Data Set Shape: {}\n' .format(conv_full_data_X.shape))

    print('Capsule Shuffled Data:')    
    caps_full_data_X, caps_full_data_y = load_images.convert_full_data_set(CAP_IMG_SIZE, 1)
    print('Capsule Full Data Set Shape: {}\n' .format(caps_full_data_X.shape))


    ## Save the convolutional label and feature info to an output folder
    conv_feature_name = 'conv_full_data_features.pt'
    conv_feature_path = f"Classification/code/Modular/Output/Processed_Data/{conv_feature_name}" 
    torch.save(conv_full_data_X, conv_feature_path)

    conv_label_name = 'conv_full_data_labels.pt'
    conv_label_path = f"Classification/code/Modular/Output/Processed_Data/{conv_label_name}" 
    torch.save(conv_full_data_y, conv_label_path)

    ## Save the capsule label and feature info to an output folder
    cap_feature_name = 'cap_full_data_features.pt'
    cap_feature_path = f"Classification/code/Modular/Output/Processed_Data/{cap_feature_name}" 
    torch.save(caps_full_data_X, cap_feature_path)

    cap_label_name = 'cap_full_data_labels.pt'
    cap_label_path = f"Classification/code/Modular/Output/Processed_Data/{cap_label_name}" 
    torch.save(caps_full_data_y, cap_label_path)



def initialize_convolutional_model():

    # load the convolutional label and feature data - 
    conv_feature_data = 'conv_full_data_features.pt'
    conv_label_data = 'conv_full_data_labels.pt'

    # Define Paramaters to Initialize Keras RandSearch Tuner Object
    Tuner_Dict = dict({'objective' : 'val_accuracy'
                       ,'max_trials' : 4
                       ,'executions_per_trial' : 1
                       ,'directory' : LOG_DIRECTORY
                       })

    conv_model = Convolutional_NN(conv_feature_data, conv_label_data, Tuner_Dict)

    # print out the tuner search space
    search_summary = conv_model.get_tuner().search_space_summary(extended=False)
    print(f'RandSearch Space: {search_summary}')

    # scale features w/o data leakage and one hot encode the label data
    conv_X_train, conv_X_test = conv_model.get_features()
    conv_y_train, conv_y_test = conv_model.get_labels()


    # run the model search hyperparameter tuner
    tuner_output_dir = 'Classification/code/Modular/Output/Conv_Tuner/'
    Search_Dict = dict({'epochs' : 3
                       ,'batch_size' : 40
                       ,'tuner_directory' : tuner_output_dir
                       ,'tuner_name' : 'tuner_00'
                       })

    # find the best current model and save the tuner object and logs to output files
    best_iteration_params = conv_model.tuner_search(Search_Dict)
    print(f'Convolutional Model Best Params: {best_iteration_params}')



def train_convolutional_model():

    import tensorflow as tf

    ## Initialize the Convolutional Model - 

    # set algorithm checkpointing and tensorboard callbacks
    best_model = tf.keras.callbacks.ModelCheckpoint(CONV_CHECKPT_LOG_DIR, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    tensorboard = tf.keras.callbacks.TensorBoard(CONV_CHECKPT_LOG_DIR, histogram_freq=0, write_graph=True, write_images=True)    
    callbacks = [best_model, tensorboard]

    # load the saved hyperparameter tuner object
    tuner_name = 'tuner_00'
    tuner = pickle.load(open(f"Classification/code/Modular/Output/Conv_Tuner/{tuner_name}.pkl","rb"))
    best_params = tuner.get_best_hyperparameters()[0].values
    print(f'\nBest Model Params: {best_params}\n')

    # reload the convolutional label and feature data 
    conv_feature_data = 'conv_full_data_features.pt'
    conv_label_data = 'conv_full_data_labels.pt'
    conv_model = Convolutional_NN(conv_feature_data, conv_label_data)

    # get scaled features w/o data leakage and one hot encoded label data
    conv_X_train, conv_X_test = conv_model.get_features()
    conv_y_train, conv_y_test = conv_model.get_labels()



    ## Train the Best Tuned Model -

    # get best convolutional model from the tuner object
    # models = tuner.get_best_models(num_models=1)
    # best_model = models[0]

    # model Name
    Mod_Num = 'Mod_001'

    # load the previosuly saved model
    best_model = keras.models.load_model(f'Classification/code/Modular/Output/Conv_Model/{Mod_Num}_cnn_tumor_brain_scan.model')

    # fit data to convolutional model
    num_epochs = 40
    history = best_model.fit(x = conv_X_train, y = conv_y_train, validation_data=(conv_X_test, conv_y_test), callbacks=callbacks, class_weight=conv_model.get_weights(), epochs=num_epochs)

    # model summary
    summary = best_model.summary()
    print('\nFit Model Summary: {summary}\n')

    # store trained convolutional model and weights
    best_model.save(f'Classification/code/Modular/Output/Conv_Model/{Mod_Num}_cnn_tumor_brain_scan.model')


    ## Evaluate the Model - 

    # test accuracy
    score = best_model.evaluate(conv_X_test, conv_y_test, verbose=0)
    print(f'\nTest loss: {score[0]} / Test accuracy: {score[1]}\n')

    # make predictions
    prediction = best_model.predict([conv_X_test]) 
    print('{} predictions: {}'.format(Mod_Num, prediction))

    # list metrics returned from callback function
    print('\ncallback function keys: {}\n' .format(history.history.keys()))

    # plot loss metric - plt.figure()
    output = 'Classification/code/Modular/Output/Model_Evaluation/Convolutional/loss_metric.png'
    plt.plot(history.history['loss'], '--')
    plt.plot(history.history['val_loss'], '--')
    plt.title('{} Model loss per epoch'.format(Mod_Num))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'evaluation'])  
    plt.savefig(output)
    plt.clf()

    # plot accuracy metric
    output = 'Classification/code/Modular/Output/Model_Evaluation/Convolutional/accuracy_metric.png'
    plt.plot(history.history['accuracy'], '--')
    plt.plot(history.history['val_accuracy'], '--')
    plt.title('{} Model accuracy per epoch'.format(Mod_Num))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'evaluation'])  
    plt.savefig(output)
    plt.clf()

    # get raw test labels
    y_train, y_test = conv_model.get_raw_labels()

    # print confusion matrix
    y_labels = []
    [y_labels.append(np.argmax(prediction[i])) for i in range(len(prediction))]
    con_mat = confusion_matrix(y_test, y_labels)
    con_df = pd.DataFrame(data=con_mat, index=CATEGORIES, columns=CATEGORIES)
    print('\nConfusion Matrix: {}\n' .format(con_df))

    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_test, y_labels)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_test, y_labels, average='weighted')
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_test, y_labels, average='weighted')
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test, y_labels, average='weighted')
    print('F1 score: %f' % f1)




def run_capsule_model():

    # load the capsule label and feature data 
    cap_feature_data = 'cap_full_data_features.pt'
    cap_label_data = 'cap_full_data_labels.pt'

    restore = True
    chkpt_dir = CAP_CHECKPT_LOG_DIR
    img_size = CAP_IMG_SIZE

    cap_model = Capsule_NN(cap_feature_data, cap_label_data, restore, chkpt_dir, img_size)
    
    # training loop parameters
    num_epochs = 5
    batch_load = 50

    cap_model.evaluate(num_epochs, batch_load)






# prepare_model_data()
# initialize_convolutional_model()
# train_convolutional_model()
run_capsule_model()



















