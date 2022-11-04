import torch
import pandas as pd
import tensorflow as tf
import numpy as np
import keras
import kerastuner
import pickle

from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from kerastuner.engine.hyperparameters import HyperParameters
from kerastuner.tuners import RandomSearch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight



def conv_model_tuner(hp):

    t_model = tf.keras.Sequential()
    t_model.add(tf.keras.layers.Input(shape=(256,256,1)))
    t_model.add(tf.keras.layers.Conv2D(hp.Int("input_units", min_value=32, max_value=64, step=32), (4, 4), padding='same', activation='relu'))
    t_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
    t_model.add(tf.keras.layers.Dropout(0.1))
    t_model.add(tf.keras.layers.Conv2D(hp.Int("middle_units_1", 64, 128, 32), (3, 3), padding='same', activation='relu'))
    t_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
    t_model.add(tf.keras.layers.Dropout(0.1))
    t_model.add(tf.keras.layers.Conv2D(hp.Int("middle_units_2", 64, 128, 32), (2, 2), padding='same', activation='relu'))
    t_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
    t_model.add(tf.keras.layers.Dropout(0.2))
    t_model.add(tf.keras.layers.Conv2D(hp.Int("middle_units_3", 96, 160, 32), (2, 2), padding='same', activation='relu'))
    t_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
    t_model.add(tf.keras.layers.Dropout(0.3))
    t_model.add(tf.keras.layers.Flatten())
    t_model.add(tf.keras.layers.Dense(hp.Int("output_units", 128, 512, 32), activation='relu'))
    t_model.add(tf.keras.layers.Dropout(0.3))
    t_model.add(tf.keras.layers.Dense(4, activation='softmax'))

    # compile the model
    opt = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.999, learning_rate=hp.Float('learning_rate'
                                                                            ,min_value=0.0005
                                                                            ,max_value=0.001
                                                                            ,default=0.001))
                                                                
    t_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return t_model



class Convolutional_NN:


    def __init__(self, features_path, labels_path, tuner_params = {}, test_size=0.2, rand_state=25):

        self.features_path = features_path
        self.features_filepath = f"Classification/code/Modular/Output/Processed_Data/{self.features_path}" 
        self.feature_data = torch.load(self.features_filepath)
        self.labels_path = labels_path
        self.labels_filepath = f"Classification/code/Modular/Output/Processed_Data/{self.labels_path}"
        self.labels_data = torch.load(self.labels_filepath)
        self.class_weights = self.compute_weights()
        self.test_size = test_size
        self.rand_state = rand_state
        if tuner_params:
            self.tuner = RandomSearch(hypermodel = conv_model_tuner
                                      ,objective = tuner_params['objective']
                                      ,max_trials = tuner_params['max_trials']
                                      ,executions_per_trial = tuner_params['executions_per_trial']
                                      ,directory = tuner_params['directory'])
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data() 
        self.X_scaled_train, self.X_scaled_test, self.y_enc_train, self.y_enc_test = self.create_processed_data() 

        self.X_scaled_train_df = pd.DataFrame()
        self.X_scaled_test_df = pd.DataFrame()

        print('Feature Data Shape: {}' .format(self.feature_data.shape))
        print('Training Data Shape: {}' .format(self.X_train.shape))



    def tuner_search(self, param_dict):

        # run the tuner search object
        self.tuner.search(x = self.X_scaled_train
                          ,y = self.y_enc_train
                          ,epochs = param_dict['epochs']
                          ,batch_size = param_dict['batch_size']
                          ,steps_per_epoch = self.X_scaled_train.shape[0] // param_dict['batch_size']
                          ,validation_data = (self.X_scaled_test, self.y_enc_test)
                          ,class_weight = self.class_weights)


        # save the tuner object to an output file
        output = param_dict['tuner_directory'] + param_dict['tuner_name'] + '.pkl'
        with open(output, "wb") as f:
            pickle.dump(self.tuner, f)


        return self.tuner.get_best_hyperparameters()[0].values


    def create_processed_data(self):

        # Set the data type
        X_train_data = self.X_train.astype('float32') 
        X_test_data = self.X_test.astype('float32')

        # Apply scaling being careful to avoid adding any data leakage into the test set
        sc = StandardScaler()

        # Reshape the training data for the scaling transform
        X_train_shape = X_train_data.shape

        X_train_dimlen_1 = X_train_shape[0]
        X_train_dimlen_2 = X_train_shape[1]
        X_train_dimlen_3 = X_train_shape[2]
        X_train_dimlen_4 = X_train_shape[3]

        X_train_two_dim = self.X_train.reshape(X_train_dimlen_1 * X_train_dimlen_2 * X_train_dimlen_3, 1) 
        
        # Scale the training data
        sc_X_train = sc.fit_transform(X_train_two_dim)

        ## Converting from array to dataframe for detailed observation
        self.X_scaled_train_df = pd.DataFrame(data = sc_X_train)

        # Reshape the data back to original format for the model
        sc_X_train = sc_X_train.reshape(X_train_dimlen_1, X_train_dimlen_2, X_train_dimlen_3, X_train_dimlen_4)
   
        ## Mapping learnt on the continuous features
        sc_map = {'mean':sc.mean_, 'std':np.sqrt(sc.var_)}

        ## Scaling the test set by transforming the mapping obtained through the training set
        X_test_shape = X_test_data.shape

        X_test_dimlen_1 = X_test_shape[0]
        X_test_dimlen_2 = X_test_shape[1]
        X_test_dimlen_3 = X_test_shape[2]
        X_test_dimlen_4 = X_test_shape[3]

        # Reshape the data for the tranform
        X_test_padded = np.zeros((X_test_dimlen_1, X_test_dimlen_2, X_test_dimlen_3, X_test_dimlen_4))
        X_test_two_dim = self.X_test.reshape(X_test_dimlen_1 * X_test_dimlen_2 * X_test_dimlen_3, X_test_dimlen_4) 
        
        # Scale the test data using the mapping
        sc_X_test = sc.transform(X_test_two_dim)

        ## Converting test arrays to dataframes for closer inspection
        self.X_scaled_test_df = pd.DataFrame(data = sc_X_test)

        # Reshape the data back to the original format for the model
        sc_X_test = sc_X_test.reshape(X_test_dimlen_1, X_test_dimlen_2, X_test_dimlen_3, X_test_dimlen_4)

        # One hot encode the labels for the model
        enc_y_train = tf.keras.utils.to_categorical(self.y_train)
        enc_y_test = tf.keras.utils.to_categorical(self.y_test)

        return sc_X_train, sc_X_test, enc_y_train, enc_y_test

    def compute_weights(self):

        # Set the training class weight proportions in case the data could not be balanced using the augmentation step
        class_weights = {}
        weight_classes = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(self.labels_data), y=self.labels_data)
        for i in range(len(weight_classes)):
            class_weights.update({i:weight_classes[i]})

        return class_weights

    def split_data(self):

        X_train, X_test, y_train, y_test = train_test_split(self.feature_data, self.labels_data, test_size = self.test_size, random_state = self.rand_state)
        return X_train, X_test, y_train, y_test

    def get_raw_features(self):
        return self.X_train, self.X_test

    def get_raw_labels(self):
        return self.y_train, self.y_test  

    def get_features(self):
        return self.X_scaled_train, self.X_scaled_test

    def get_labels(self):
        return self.y_enc_train, self.y_enc_test  

    def get_weights(self):
        return self.class_weights

    def get_tuner(self):
        return self.tuner

    def get_best_params(self):
        return self.best_model_params