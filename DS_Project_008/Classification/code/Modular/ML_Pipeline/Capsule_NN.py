import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix

import tensorflow.compat.v1 as tf_
tf_.disable_v2_behavior() 
tf_.reset_default_graph()
np.random.seed(42)
tf_.set_random_seed(42)

module_path = Path(__file__).parents[1]
sys.path.append(str(module_path))

from ML_Pipeline.Config import *


# Create normalized output vectors, Epsilon value is designed to avoid a 'divide by zero' error
def squash_fn(s, axis=-1, epsilon=1e-7, name=None):
    with tf_.name_scope(name, default_name="squash_fn"):
        squared_norm = tf_.reduce_sum(tf_.square(s), axis=axis, keepdims=True)
        safe_norm = tf_.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm

    return squash_factor * unit_vector


# Normalize output of the output label vectors (avoid 'divide by zero' error)
def find_norm(s, axis=-1, epsilon=1e-7, keepdims=False, name=None):
    with tf_.name_scope(name, default_name="safe_norm"):
        squared_norm = tf_.reduce_sum(tf_.square(s), axis=axis, keepdims=keepdims)
    
    return tf_.sqrt(squared_norm + epsilon)


def create_conv_layer_tensors(self):

    # Define the convolutional layers that will feed into the primary capsule layer
    conv_layer1_params = {
        "filters": self.conv_resize,
        "kernel_size": self.kernel_size, 
        "strides": 1,
        "padding": "valid",
        "activation": tf_.nn.relu,
    }
    self.conv_layer1_tensor = \
        tf_.layers.conv2d(self.X_tensor,
                          name="conv_layer1_tensor",
                          **conv_layer1_params)

    conv_layer2_params = {
        "filters": self.layer1_num_maps * self.layer1_num_dimensions,
        "kernel_size": self.kernel_size, 
        "strides": 2,
        "padding": "valid",
        "activation": tf_.nn.relu
    }
    self.conv_layer2_tensor = \
        tf_.layers.conv2d(self.conv_layer1_tensor,
                          name="conv_layer2_tensor",
                          **conv_layer2_params)

    # Flattened raw output of the primary capsule layer tensors
    layer1_raw_output_tensor = \
        tf_.reshape(self.conv_layer2_tensor,
                    [-1, self.layer1_num_caps,
                    self.layer1_num_dimensions],
                    name="layer1_raw_output_tensor")

    # Squash the capsule output tensor (normalize)
    self.layer1_output_tensor = \
        squash_fn(layer1_raw_output_tensor, name="layer1_output_tensor")


def create_weight_tensor(self):

    init_sigma = 0.1
    Weight_init = \
        tf_.random_normal(
        shape=(1, self.layer1_num_caps, 
               self.layer2_num_caps,
               self.layer2_num_dimensions,
               self.layer1_num_dimensions),
        stddev=init_sigma, dtype=tf_.float32, name="Weight_init")

    self.weights_tensor = tf_.Variable(Weight_init, name="Weights")


def create_layer1_output_tensor(self):

    layer1_output_expanded = \
        tf_.expand_dims(self.layer1_output_tensor, -1, name="layer1_output_expanded")
    layer1_output_tile = \
        tf_.expand_dims(layer1_output_expanded, 2, name="layer1_output_tile")

    return  tf_.tile(layer1_output_tile,
                     [1, 1, self.layer2_num_caps, 1, 1],
                     name="layer1_output_tensor") 


def create_routing_weights_round1_sum_tensor(self):

    # multiply the routing weights by the predicted capsules
    weighted_predictions_round1 = \
        tf_.multiply(self.routing_weights_round1_tensor,
                     self.layer2_prediction_tensor,
                     name="weighted_predictions_round1")

    self.routing_weights_round1_sum_tensor = \
        tf_.reduce_sum(weighted_predictions_round1,
                       axis=1,
                       keepdims=True,
                       name="routing_weights_round1_sum_tensor")


def create_RBA_round1_tensor(self):

    layer2_round1_output_tiled = \
        tf_.tile(self.layer2_round1_output_tensor,
                 [1, self.layer1_num_caps, 1, 1, 1],
                 name="layer2_round1_output_tiled") 

    self.routing_agreement_tensor = \
        tf_.matmul(self.layer2_prediction_tensor,
                   layer2_round1_output_tiled,
                   transpose_a=True,
                   name="routing_agreement_tensor")


def create_round2_routing_weights_tensor(self):

    RBA_round2_weights = \
        tf_.add(self.routing_weights_round1_tensor,
                self.routing_agreement_tensor,
                name="RBA_round2_weights")

    self.routing_weights_round2_tensor = \
        tf_.nn.softmax(RBA_round2_weights,
                       axis=2,
                       name="routing_weights_round2_tensor")


def create_routing_weights_round2_sum_tensor(self):

    weighted_predictions_round2 = \
        tf_.multiply(self.routing_weights_round2_tensor,
                     self.layer2_prediction_tensor,
                     name="weighted_predictions_round2")

    return tf_.reduce_sum(weighted_predictions_round2,
                          axis=1,
                          keepdims=True,
                          name="routing_weights_round2_sum_tensor")


def compute_label_prediction_tensor(self):

    y_proba_tensor = \
        find_norm(self.layer2_output_tensor, axis=-2, name="y_proba_tensor")

    y_proba_argmax = tf_.argmax(y_proba_tensor, axis=2, name="y_proba_argmax")

    # remove unnecessary dimensions
    self.y_pred_tensor = \
        tf_.squeeze(y_proba_argmax,
                    axis=[1, 2],
                    name="y_pred_tensor")


def create_margin_loss_compute_tensor(self):

    # Margin loss constants
    m_plus = 0.9
    m_minus = 0.1
    lambda_ = 0.5

    layer2_output_tensor_norm = \
        find_norm(self.layer2_output_tensor,
                  axis=-2,
                  keepdims=True,
                  name="layer2_output_tensor_norm")

    # Compute margin loss for each capsule.
    present_error_computation_tensor = \
        tf_.square(tf_.maximum(0., m_plus - layer2_output_tensor_norm),
                   name="present_error_computation_tensor")

    present_error_tensor = \
        tf_.reshape(present_error_computation_tensor,
                    shape=(-1, self.layer2_num_caps),
                    name="present_error_tensor") 

    # Downweighting absent entity loss to stop shrinking activity vector lengths during learning
    absent_error_computation_tensor = \
        tf_.square(tf_.maximum(0., layer2_output_tensor_norm - m_minus),
                   name="absent_error_computation_tensor")

    absent_error_tensor = \
        tf_.reshape(absent_error_computation_tensor,
                    shape=(-1, self.layer2_num_caps),
                    name="absent_error_tensor")

    # Compute the margin loss value for each instance multiplied by each label
    combined_margin_loss_tensor = \
        tf_.add(self.y_tensor_onehot * present_error_tensor,
                lambda_ * (1.0 - self.y_tensor_onehot) * absent_error_tensor,
                name="combined_margin_loss_tensor")

    self.margin_loss_computation_tensor = \
        tf_.reduce_mean(tf_.reduce_sum(combined_margin_loss_tensor, axis=1),
                        name="margin_loss_computation_tensor")


def create_decoder_reconstruction_input_tensor(self):
    
    # Use ground truths for training, o/w prediction values for 
    # the unsupervised learning decoder reconstruction
    reconstruction_values_tensor = \
        tf_.cond(self.reconstruction_label_mask_tensor,
                 lambda: self.y_tensor,
                 lambda: tf_.cast(self.y_pred_tensor, dtype=tf_.int64),
                 name="reconstruction_values_tensor")    

    # One hot encode the reconstruction mask
    reconstruction_mask_tensor = \
        tf_.one_hot(reconstruction_values_tensor,
                    depth=self.layer2_num_caps,
                    name="reconstruction_mask_tensor")

    # Re-shape the reconstruction mask by increasing the rank in order 
    # to make the multiplication possible
    reconstruction_mask_reshaped_tensor = \
        tf_.reshape(reconstruction_mask_tensor,
                    [-1, 1, self.layer2_num_caps, 1, 1],
                    name="reconstruction_mask_reshaped_tensor")
    
    # Create a variable to hold the masked output label predictions
    layer2_output_masked_tensor = \
        tf_.multiply(self.layer2_output_tensor,
                     reconstruction_mask_reshaped_tensor,
                     name="layer2_output_masked_tensor")

    self.decoder_input_tensor = \
        tf_.reshape(layer2_output_masked_tensor,
                    [-1, self.layer2_num_caps * self.layer2_num_dimensions],
                    name="decoder_input_tensor")


def create_reconstruction_loss_tensor(self):

    # The Decoder is a 3 Layer Dense Network 
    layer1_num_hidden_units = 1024 # 512
    layer2_num_hidden_units = 2048 # 1024
    output_layer_num_units = (self.cap_resize)**2  ## todo: Automate input sizes

    with tf_.name_scope("decoder"):
        decoder_hidden_layer1_tensor = \
            tf_.layers.dense(self.decoder_input_tensor,
                             layer1_num_hidden_units,
                             activation=tf_.nn.relu,
                             name="decoder_hidden_layer1_tensor")

        decoder_hidden_layer2_tensor = \
            tf_.layers.dense(decoder_hidden_layer1_tensor,
                             layer2_num_hidden_units,
                             activation=tf_.nn.relu,
                             name="decoder_hidden_layer2_tensor")

        decoder_output_layer_tensor = \
            tf_.layers.dense(decoder_hidden_layer2_tensor,
                             output_layer_num_units,
                             activation=tf_.nn.sigmoid,
                             name="decoder_output_layer_tensor")

    # Mean squared difference between the input values
    X_tensor_flattened_tensor = \
        tf_.reshape(self.X_tensor,
                    [-1, output_layer_num_units],
                    name="X_tensor_flattened_tensor")

    squared_difference_tensor = \
        tf_.square(X_tensor_flattened_tensor - decoder_output_layer_tensor,
                   name="squared_difference_tensor")

    self.reconstruction_loss_tensor = \
        tf_.reduce_mean(squared_difference_tensor,
                        name="reconstruction_loss_tensor")


def compute_combined_loss_tensor(self):

    alpha = 0.0005
    self.combined_loss_tensor = \
        tf_.add(self.margin_loss_computation_tensor,
                alpha * self.reconstruction_loss_tensor,
                name="combined_loss_tensor")


def create_prediction_tensors(self):

    # Create a variable that will hold all of the trainable label prediction transformation matrices
    self.create_weight_tensor()

    # Create an array with one identical weight matrix variable per instance in the data set
    self.batch_size_tensor = tf_.shape(self.X_tensor)[0]
    tiled_weights_tensor = tf_.tile(self.weights_tensor,
                                    [self.batch_size_tensor, 1, 1, 1, 1],
                                    name="tiled_weights_tensor")

    # Create a variable that will hold the capsule predictions for each label per batch
    layer1_output_tensor = self.create_layer1_output_tensor()

    # Multiply the capsule outputs with the weight matrices to get all of the label predictions
    self.layer2_prediction_tensor = \
        tf_.matmul(tiled_weights_tensor,
                   layer1_output_tensor,
                   name="layer2_prediction_tensor")


def create_RBA1_tensors(self):

    # Create the 1st round routing weights tensor variable
    RBA_round1_weights = tf_.zeros([self.batch_size_tensor, self.layer1_num_caps, self.layer2_num_caps, 1, 1],
                                    dtype=np.float32,
                                    name="RBA_round1_weights")
    self.routing_weights_round1_tensor = \
        tf_.nn.softmax(RBA_round1_weights, axis=2, name="routing_weights_round1_tensor")

    # Create a tensor variable to sum the routing weights
    self.create_routing_weights_round1_sum_tensor()

    # Create a normalized vector for the weighted sum output of the 1st round of routing weights
    self.layer2_round1_output_tensor = \
        squash_fn(self.routing_weights_round1_sum_tensor,
                  axis=-2,
                  name="layer2_round1_output_tensor")

    # Create a variable to hold the scalar product for 1st round of routing by agreement
    self.create_RBA_round1_tensor()


def create_RBA2_tensors(self):

    # Create the 2nd round routing weights tensor variable - routing_weights_round1
    self.create_round2_routing_weights_tensor()

    # Create a tensor variable to sum the round 2 routing weights
    routing_weights_round2_sum_tensor = self.create_routing_weights_round2_sum_tensor()

    # Create a variable to hold the round 2 RBA capsule output vectors
    self.layer2_output_tensor = \
        squash_fn(routing_weights_round2_sum_tensor, axis=-2, name="layer2_output_tensor")


def create_RBA_computation_tensors(self):

    self.create_RBA1_tensors()
    self.create_RBA2_tensors()


def create_prediction_loss_tensors(self):

    # Create a variable to compute the class prediction for each image in the current batch
    self.compute_label_prediction_tensor()

    # Create a placeholder variable for the output labels in one hot encoded format
    self.y_tensor = tf_.placeholder(shape=[None], dtype=tf_.int64, name="y_tensor")

    self.y_tensor_onehot = \
        tf_.one_hot(self.y_tensor,
                   depth=self.layer2_num_caps,
                   name="y_tensor_onehot")

    # Create a variable to calculate the margin loss for each predicted label
    self.create_margin_loss_compute_tensor()


def create_reconstruction_decoder_tensors(self):

    # Create a mask variable - array value is 1.0 for the target class, otherwise 0.0  
    self.reconstruction_label_mask_tensor = \
        tf_.placeholder_with_default(False,
                                     shape=(),
                                     name="reconstruction_label_mask_tensor")

    # Create a variable to hold all of the flattened reconstruction decoder inputs        
    self.create_decoder_reconstruction_input_tensor()

    # Create a variable to calculate the reconstruction loss, help with overfitting
    self.create_reconstruction_loss_tensor()


def create_utility_tensors(self):

    # Create a loss variable to hold the weighted sum of the margin and reconstruction losses
    self.compute_combined_loss_tensor()

    self.correct_tensor = \
        tf_.equal(self.y_tensor,
                  tf_.cast(self.y_pred_tensor, dtype=tf_.int64),
                  name="correct_tensor")

    self.accuracy_tensor = \
        tf_.reduce_mean(tf_.cast(self.correct_tensor, tf_.float32),
                        name="accuracy_tensor")
    
    tensor_optimizer_object = tf_.train.AdamOptimizer()

    self.tensor_training_optimizer = \
        tensor_optimizer_object.minimize(self.combined_loss_tensor,
                                         name="tensor_training_optimizer")

    self.initializer_object = tf_.global_variables_initializer()
    self.saver_object = tf_.train.Saver()


def create_capsule_network_tensors(self):

    self.X_tensor = \
        tf_.placeholder(shape=[None, self.cap_resize, self.cap_resize, 1],
        dtype=tf_.float32, name="X_tensor") 

    self.create_conv_layer_tensors()
    self.create_prediction_tensors()
    self.create_RBA_computation_tensors()  # routing by agreement
    self.create_prediction_loss_tensors()
    self.create_reconstruction_decoder_tensors()
    self.create_utility_tensors()


def define_capsule_network_variables(self):

    self.cap_features_path = \
        os.path.join(module_path,
                     "Output/Processed_Data/cap_full_data_features.pt")
    self.cap_feature_data = torch.load(self.cap_features_path)
    self.cap_labels_path = \
        os.path.join(module_path,
                     "Output/Processed_Data/cap_full_data_labels.pt")
    self.cap_label_data = torch.load(self.cap_labels_path)

    self.layer1_num_maps = 32
    self.layer1_num_dimensions = 8
    self.layer2_num_dimensions = 16
    self.kernel_size = 9
    self.num_filters = self.layer1_num_maps * self.layer1_num_dimensions
    self.layer1_num_caps = \
        np.int64((((self.cap_resize - (2 * (self.kernel_size - 1))) / 2) ** 2) * self.layer1_num_maps)
    self.layer2_num_caps = len(IMAGE_FOLDERS)


def init_cap_model(self):

    self.define_capsule_network_variables()
    self.create_capsule_network_tensors()   


def evaluate_cap_model(self, model_folder_name=None):

    num_epochs = 1
    batch_size = 50
    save_dir = \
        os.path.join(module_path,
                     f"Output/Checkpointing/{model_folder_name}/Capsule") # "Capsule"

    train_len = len(self.cap_X_train_scaled)
    test_len = len(self.cap_X_test_scaled)
    val_len = len(self.cap_X_val_scaled)
    num_iterations_per_training_epoch = train_len // batch_size
    num_iterations_test = test_len // batch_size
    num_iterations_validation = val_len // batch_size
    best_loss_val = np.infty

    #Training loop body
    with tf_.Session() as sess:

        if model_folder_name and tf_.train.checkpoint_exists(save_dir):
            self.saver_object.restore(sess, save_dir)

        else:
            self.initializer_object.run()

        for epoch in range(num_epochs):

            for iteration in range(1, (num_iterations_per_training_epoch + 2)):

                current_marker = iteration * batch_size
                start = (iteration - 1) * batch_size
                stop = \
                    current_marker if (current_marker <= train_len) else start + (batch_size - (current_marker - train_len)) + 1
                X_batch, y_batch = \
                    self.cap_X_train_scaled[start:stop], self.cap_y_train[start:stop]

                # Run the training operation and measure the loss:
                _, loss_train = \
                    sess.run([self.tensor_training_optimizer, self.combined_loss_tensor],
                             feed_dict={self.X_tensor: X_batch.reshape([-1, self.cap_resize, self.cap_resize, 1]),
                                        self.y_tensor: y_batch,
                                        self.reconstruction_label_mask_tensor: True})

                print(f"\rIteration: {iteration}/{num_iterations_per_training_epoch} "
                      f"({iteration * 100 / num_iterations_per_training_epoch:.1f}%)  "
                      f"Loss: {loss_train:.5f}", end="")

            # Measuring the validation loss and accuracy at the end of each epoch
            loss_vals = []
            acc_vals = []
            for iteration in range(1, (num_iterations_validation + 2)):

                current_marker = iteration * batch_size
                start = (iteration - 1) * batch_size
                stop = \
                    current_marker if (current_marker <= val_len) else start + (batch_size - (current_marker - val_len)) + 1
                X_batch, y_batch = self.cap_X_val_scaled[start:stop], self.cap_y_val[start:stop]   
                loss_val, acc_val = \
                    sess.run([self.combined_loss_tensor, self.accuracy_tensor],
                             feed_dict={self.X_tensor: X_batch.reshape([-1, self.cap_resize, self.cap_resize, 1]),
                                        self.y_tensor: y_batch})

                loss_vals.append(loss_val)
                acc_vals.append(acc_val)
                
                print(f"\rEvaluating the model: {iteration}/{num_iterations_validation} "
                      f"({iteration * 100 / num_iterations_validation:.1f}%)", end=" " * 10)

            loss_val = np.mean(loss_vals)
            acc_val = np.mean(acc_vals)

            print(f"\rEpoch: {epoch + 1}  Val accuracy: {acc_val * 100:.4f}%  "
                  f"Loss: {loss_val:.6f}"
                  f"{' (improved)' if loss_val < best_loss_val else ''}", end="")

            # And save the model if it improved:
            if loss_val < best_loss_val:
                save_path = self.saver_object.save(sess, save_dir)
                best_loss_val = loss_val

    # Evaluate Model Performance
    with tf_.Session() as sess:

        self.saver_object.restore(sess, save_dir)

        loss_tests = []
        acc_tests = []
        for iteration in range(1, (num_iterations_test + 2)):

            current_marker = iteration * batch_size
            start = (iteration - 1) * batch_size
            stop = \
                current_marker if (current_marker <= test_len) else start + (batch_size - (current_marker - test_len)) + 1
            X_batch, y_batch = self.cap_X_test_scaled[start:stop], self.cap_y_test[start:stop]   
            loss_test, acc_test = \
                sess.run([self.combined_loss_tensor, self.accuracy_tensor],
                         feed_dict={self.X_tensor: X_batch.reshape([-1, self.cap_resize, self.cap_resize, 1]), 
                                    self.y_tensor: y_batch})                
            loss_tests.append(loss_test)
            acc_tests.append(acc_test)

            print(f"\rEvaluating the model: {iteration}/{num_iterations_test} "
                  f"({iteration * 100 / num_iterations_test:.1f}%)", end=" " * 10)


        loss_test = np.mean(loss_tests)
        acc_test = np.mean(acc_tests)
        print("\rFinal test accuracy: {:.4f}%  Loss: {:.6f}".format(acc_test * 100, loss_test))


    batch_labels = []
    with tf_.Session() as sess:

        self.saver_object.restore(sess, save_dir)

        for iteration in range(1, (num_iterations_test + 2)):

            # Handles final batch being a different size than previous iterations with null padding
            current_marker = iteration * batch_size
            start = (iteration - 1) * batch_size
            stop = \
                current_marker if (current_marker <= test_len) else start + (batch_size - (current_marker - test_len)) + 1
            X_batch, y_batch = self.cap_X_test_scaled[start:stop], self.cap_y_test[start:stop]
            result_labels = (sess.run([self.y_pred_tensor],
                                         feed_dict={self.X_tensor: X_batch.reshape([-1, self.cap_resize, self.cap_resize, 1]),
                                                    self.y_tensor: y_batch}))
            null_padded_labels = np.full((batch_size,), np.nan)
            null_padded_labels[:len(result_labels[0])] = result_labels[0]
            batch_labels.append(null_padded_labels)

        labels = np.array(batch_labels)
        label_preds = np.concatenate([arr.flatten() for arr in labels])
        self.cap_test_prediction_list = label_preds[~np.isnan(label_preds)]

        con_mat = confusion_matrix(self.cap_y_test, self.cap_test_prediction_list)
        confusion_matrix_df = \
            pd.DataFrame(data=con_mat, index=IMAGE_FOLDERS, columns=IMAGE_FOLDERS)

        print(confusion_matrix_df, '\n')
        self.print_classification_scores('cap')

    ''' ## todo: add decoder plots
    n_samples = 5
    sample_images = cap_X_test[:n_samples].reshape([-1, CAP_IMG_RESIZE, CAP_IMG_RESIZE, 1])
    with tf_.Session() as sess:
        saver.restore(sess, CAP_CHECKPT_LOG_DIR)
        caps2_output_value, decoder_output_value, y_pred_value = sess.run([caps2_output, decoder_output, y_pred],
                                                                          feed_dict={X: sample_images, y: np.array([], dtype=np.int64)})

    sample_images = sample_images.reshape(-1, CAP_IMG_RESIZE, CAP_IMG_RESIZE)
    reconstructions = decoder_output_value.reshape([-1, CAP_IMG_RESIZE, CAP_IMG_RESIZE])

    plt.figure(figsize=(n_samples * 2, 3))
    for index in range(n_samples):
        plt.subplot(1, n_samples, index + 1)
        plt.imshow(sample_images[index], cmap="binary")
        plt.title("Label:" + str(cap_y_test[index]))
        plt.axis("off")

    plt.show()

    plt.figure(figsize=(n_samples * 2, 3))
    for index in range(n_samples):
        plt.subplot(1, n_samples, index + 1)
        plt.title("Predicted:" + str(y_pred_value[index]))
        plt.imshow(reconstructions[index], cmap="binary") #, vmin=0, vmax=1)
        plt.axis("off")'''
