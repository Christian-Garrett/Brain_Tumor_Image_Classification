import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
import cloudpickle as pickle
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from keras_tuner.tuners import RandomSearch  #.tuners

module_path = Path(__file__).parents[1]
sys.path.append(str(module_path))

from ML_Pipeline.Config import *


def model_tuner(hp):

    tuner_model = keras.Sequential()
    tuner_model.add(keras.layers.Input(shape=(256, 256, 1)))
    tuner_model.add(keras.layers.Conv2D(hp.Int("input_units", min_value=32, max_value=64, step=32),
                                              (4, 4),
                                              padding='same',
                                              activation='relu'))
    tuner_model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
    tuner_model.add(keras.layers.Dropout(0.1))
    tuner_model.add(keras.layers.Conv2D(hp.Int("middle_units_1", 64, 128, 32),
                                       (3, 3),
                                       padding='same',
                                       activation='relu'))
    tuner_model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
    tuner_model.add(keras.layers.Dropout(0.1))
    tuner_model.add(keras.layers.Conv2D(hp.Int("middle_units_2", 64, 128, 32),
                                       (2, 2),
                                       padding='same',
                                       activation='relu'))
    tuner_model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
    tuner_model.add(keras.layers.Dropout(0.2))
    tuner_model.add(keras.layers.Conv2D(hp.Int("middle_units_3", 96, 160, 32),
                                       (2, 2),
                                       padding='same',
                                       activation='relu'))
    tuner_model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
    tuner_model.add(keras.layers.Dropout(0.3))
    tuner_model.add(keras.layers.Flatten())
    tuner_model.add(keras.layers.Dense(hp.Int("output_units", 128, 512, 32),
                                      activation='relu'))
    tuner_model.add(keras.layers.Dropout(0.3))
    tuner_model.add(keras.layers.Dense(4, activation='softmax'))

    # compile the tuner model
    opt = keras.optimizers.Adam(beta_1=0.9,
                                beta_2=0.999,
                                learning_rate=hp.Float('learning_rate',
                                                        min_value=0.0005,
                                                        max_value=0.001,
                                                        default=0.001))
                                                                
    tuner_model.compile(optimizer=opt,
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

    return tuner_model


def get_model_summary(model):

    summary = model.summary()
    print(f'\nFit Model Summary: {summary}\n')


def get_model_metrics(self, model_):

    score = model_.evaluate(self.conv_X_test_scaled, np.asarray(self.conv_y_test_OHE), verbose=0)
    print(f'\nTest loss: {score[0]} / Test accuracy: {score[1]}\n')

    predictions = model_.predict([self.conv_X_test_scaled]) 
    self.conv_test_prediction_list = [np.argmax(predictions[i]) for i in range(len(predictions))]
    con_mat = confusion_matrix(self.conv_y_test, self.conv_test_prediction_list)
    con_df = pd.DataFrame(data=con_mat, index=IMAGE_FOLDERS, columns=IMAGE_FOLDERS)
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.width', 1000):
        print(f'\nConfusion Matrix:\n {con_df}\n')


def plot_model_checkpoints(model):  ## todo: add this function
    return None


def load_model_from_file(model_name):

    model_path = f"Output/Conv_Model/{model_name}_cnn_tumor_brain_scan.model"
    model_object = os.path.join(module_path, model_path)

    try:
        conv_model = keras.models.load_model(model_object)
        print(f"Previously saved conv model loaded successfully: {conv_model}")

    except OSError:
        print(f"Could not open/read file: {model_name}")
        sys.exit()

    return conv_model


def evaluate_conv_model(self, model_name=None):

    if(model_name):
        model = load_model_from_file(model_name)

    ## keras objects are not copying properly to class attributes -
    else:
        # define tuner model params
        tuner_search_results_dir = \
            os.path.join(module_path,
                        f"Output/Log_Directory/Convolutional/{int(time.time())}_")

        # initialize tuner object  - # 4, 1, 3
        conv_tuner = \
            RandomSearch(hypermodel=model_tuner,
                        objective='val_accuracy',
                        max_trials=1,
                        executions_per_trial=1,
                        directory=tuner_search_results_dir,
                        project_name='refactor')

        search_summary = conv_tuner.search_space_summary(extended=False)
        print(f'Hyperparameter Tuner Search Space: {search_summary}')

        # tuner search
        conv_tuner.search(x=self.conv_X_train_scaled,
                            y=self.conv_y_train_OHE,
                            epochs=1,
                            batch_size=40,
                            steps_per_epoch=self.conv_X_train_scaled.shape[0] // 40,
                            class_weight=self.class_weights,
                            validation_data=(self.conv_X_test_scaled, self.conv_y_test_OHE))
        
        # tuner search results
        self.best_conv_hyperparameters = \
            conv_tuner.get_best_hyperparameters()[0].values
        print(f'Convolutional Model Best Params: {self.best_conv_hyperparameters}')

        # define tuner search params
        tuner_model_dir = os.path.join(module_path, "Output/Conv_Tuner/")

        # save tuner obect
        output = tuner_model_dir + 'tuner_122224.pkl'
        with open(output, "wb") as f:
            pickle.dump(conv_tuner, f)

        best_params = conv_tuner.get_best_hyperparameters()[0].values
        print(f'\nBest Model Params: {best_params}\n')
        models = conv_tuner.get_best_models(num_models=1)
        model = models[0]

        model_checkpoints = \
            keras.callbacks.ModelCheckpoint(CONV_CHECKPT_LOG_DIR,
                                            monitor='val_accuracy',
                                            verbose=1,
                                            save_best_only=True,
                                            mode='max')
        tensorboard = \
            keras.callbacks.TensorBoard(CONV_CHECKPT_LOG_DIR,
                                        histogram_freq=0,
                                        write_graph=True,
                                        write_images=True)
    
        conv_model_callbacks = [model_checkpoints, tensorboard]

        num_epochs = 40
        self.conv_training_history = \
            model.fit(x=self.conv_X_train_scaled,
                      y=self.conv_y_train_OHE,
                      validation_data=(self.conv_X_test_scaled, self.conv_y_test_OHE),
                      callbacks=conv_model_callbacks,
                      class_weight=self.class_weights,
                      epochs=num_epochs)

    get_model_summary(model)
    self.get_model_metrics(model_=model)
    self.print_classification_scores('conv')

    #### plot_model_checkpoints() ####
    '''# list metrics returned from callback function
    print('\ncallback function keys: {}\n' .format(self.conv_training_history.history.keys()))

    # plot loss metric
    plt.plot(self.conv_training_history.history['loss'], '--')
    plt.plot(self.conv_training_history.history['val_loss'], '--')
    plt.title('{} Model loss per epoch'.format(Mod_Num))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'evaluation'])  
    plt.savefig(os.path.join(module_path, "Output/Model_Evaluation/Convolutional/loss_metric_.png"))
    plt.clf()

    # plot accuracy metric
    plt.plot(self.conv_training_history.history['accuracy'], '--')
    plt.plot(self.conv_training_history.history['val_accuracy'], '--')
    plt.title('{} Model accuracy per epoch'.format(Mod_Num))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'evaluation'])  
    plt.savefig(os.path.join(module_path, "Output/Model_Evaluation/Convolutional/accuracy_metric.png"))
    plt.clf()'''

    # save the model if this is a new build
    if(not model_name):
        MOD_NUM = 'Model_122324' 
        save_path = \
            os.path.join(module_path,
                            f"Output/Conv_Model/{MOD_NUM}_cnn_tumor_brain_scan.model")

        model.save(save_path)
        print("model saved successfully")
