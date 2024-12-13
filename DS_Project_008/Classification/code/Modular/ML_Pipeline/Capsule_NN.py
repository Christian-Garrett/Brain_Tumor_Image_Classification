import torch
import numpy as np
import pandas as pd

import tensorflow.compat.v1 as tf_
tf_.disable_v2_behavior() 

tf_.reset_default_graph()
np.random.seed(42)
tf_.set_random_seed(42)

from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Image folder names
CATEGORIES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]


# Create output vectors with length <= 1, Epsilon value is designed to avoid a 'divide by zero' error
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



class Capsule_NN:


    def __init__(self, features_path, labels_path, new_model, save_dir, img_dim, test_size=0.2, val_size=0.127, rand_state=25, c1_n_maps = 32, c1_n_dims = 8, c2_n_dims = 16, kernel = 9):

        self.features_path = features_path
        self.features_filepath = f"Classification/code/Modular/Output/Processed_Data/{self.features_path}" 
        self.feature_data = torch.load(self.features_filepath)
        self.labels_path = labels_path
        self.labels_filepath = f"Classification/code/Modular/Output/Processed_Data/{self.labels_path}"
        self.labels_data = torch.load(self.labels_filepath)
        self.class_weights = self.compute_weights()
        self.test_size = test_size
        self.val_size = val_size
        self.rand_state = rand_state
        self.X_train, self.X_test, self.X_val, self.y_train, self.y_test, self.y_val = self.split_data() 
        self.X_scaled_train, self.X_scaled_test, self.X_scaled_val, self.y_np_train, self.y_np_test, self.y_np_val = self.create_processed_data() 

        self.X_scaled_train_df = pd.DataFrame()
        self.X_scaled_test_df = pd.DataFrame()

        self.new_model = new_model
        self.save_dir = save_dir 

        self.img_dim = img_dim
        self.caps1_n_maps = c1_n_maps
        self.caps1_n_dims = c1_n_dims
        self.caps2_n_dims = c2_n_dims
        self.kernel_size = kernel
        self.filters = self.caps1_n_maps * self.caps1_n_dims
        self.caps1_n_caps = self.calc_caps1_caps()
        self.caps2_n_caps = len(CATEGORIES)

        ## Define the Capsule Network Variables :-

        # Create placeholder for the input images
        self.X = tf_.placeholder(shape=[None, self.img_dim, self.img_dim, 1], dtype=tf_.float32, name="X")
        # Create convolutional input layers
        self.conv1, self.conv2 = self.create_conv_layers()
        # Flattened raw output of the primary capsule layer (batch size, 2D capsule map X num capsule maps, dimension of each capsule)
        self.caps1_raw = tf_.reshape(self.conv2, [-1, self.caps1_n_caps, self.caps1_n_dims], name="caps1_raw")
        # Squash the output vectors
        self.caps1_output = squash_fn(self.caps1_raw, name="caps1_output")

        ## Computing Capsule Predictions :-

        # Create a variable that will hold all of the trainable label prediction transformation matrices
        self.Weights = self.create_weight_matrices_variable()
        # Create an array with one identical weight matrix variable per instance in the data set
        self.batch_size = tf_.shape(self.X)[0]
        self.tiled_weights = tf_.tile(self.Weights, [self.batch_size, 1, 1, 1, 1], name="tiled_weights")
        # Create a variable that will hold the capsule predictions for each label per batch
        self.caps1_output_tiled = self.create_caps1_output_variable()
        # Multiply the capsule outputs with the weight matrices to get all of the label predictions
        self.caps2_predicted = tf_.matmul(self.tiled_weights, self.caps1_output_tiled, name="caps2_predicted")

        ## Computing RBA Round 1 Variables :-

        # Create the 1st round routing weights tensor variable
        self.raw_weights = tf_.zeros([self.batch_size, self.caps1_n_caps, self.caps2_n_caps, 1, 1], dtype=np.float32, name="raw_weights")
        self.routing_weights = tf_.nn.softmax(self.raw_weights, axis=2, name="routing_weights")
        # Create a tensor variable to sum the routing weights
        self.weighted_sum = self.create_weighted_sum_variable()
        # Create a variable to hold squashed weighted sum output of the 1st round of routing weights
        self.caps2_output_round_1 = squash_fn(self.weighted_sum, axis=-2, name="caps2_output_round_1")
        # Create a variable to hold the scalar product for 1st round of routing by agreement
        self.agreement = self.create_round_1_RBA_variable()

        ## Computing RBA Round 2 Variables :-

        # Create the 2nd round routing weights tensor variable
        self.routing_weights_round_2 = self.create_round_2_routing_weights_variable()
        # Create a tensor variable to sum the round 2 routing weights
        self.weighted_sum_round_2 = self.create_weighted_sum_round_2_variable()
        # Create a variable to hold the round 2 RBA capsule output vectors
        self.caps2_output = squash_fn(self.weighted_sum_round_2, axis=-2, name="caps2_output_round_2")

        ## Output Label Prediction Loss Variables :-

        # Create a variable to compute the class prediction for each image in the current batch
        self.y_pred = self.compute_predicted_class()
        # Create a placeholder variable for the output labels in one hot encoded format
        self.y = tf_.placeholder(shape=[None], dtype=tf_.int64, name="y")
        self.T = tf_.one_hot(self.y, depth=self.caps2_n_caps, name="T")
        # Create a variable to calculate the margin loss for each predicted label
        self.margin_loss = self.compute_margin_loss()

        ## Reconstruction Decoder Variables (preserving image info to limit overfitting):- 

        # Create a mask variable - array value is 1.0 for the target class, otherwise 0.0  
        self.mask_with_labels = tf_.placeholder_with_default(False, shape=(), name="mask_with_labels")
        # Create a variable to hold all of the flattened reconstruction decoder inputs        
        self.decoder_input = self.create_decoder_input_variable()
        # Create a variable to calculate the reconstruction loss
        self.reconstruction_loss = self.create_reconstruction_loss_variable()

        ## Network Accuracy Evaluation and Training Variables :-

        # Create a loss variable to hold the weighted sum of the margin and reconstruction losses
        self.loss = self.create_loss_variable()
        self.correct = tf_.equal(self.y, self.y_pred, name="correct")
        self.accuracy = tf_.reduce_mean(tf_.cast(self.correct, tf_.float32), name="accuracy")
     
        self.optimizer = tf_.train.AdamOptimizer()
        self.training_op = self.optimizer.minimize(self.loss, name="training_op")

        self.init = tf_.global_variables_initializer()
        self.saver = tf_.train.Saver()    


    def evaluate(self, n_epochs, batch_size):

        n_iterations_per_epoch = len(self.X_scaled_train) // batch_size
        n_iterations_validation = len(self.X_scaled_val) // batch_size
        best_loss_val = np.infty

        #Training loop body
        with tf_.Session() as sess:
            if self.new_model and tf_.train.checkpoint_exists(self.save_dir):
                self.saver.restore(sess, self.save_dir)
            else:
                self.init.run()

            for epoch in range(n_epochs):
                for iteration in range(1, n_iterations_per_epoch + 1):
                    start = (iteration-1)*batch_size
                    stop = iteration*batch_size
                    X_batch, y_batch = self.X_scaled_train[start:stop], self.y_np_train[start:stop]
                    # Run the training operation and measure the loss:
                    _, loss_train = sess.run([self.training_op, self.loss]
                                             , feed_dict={self.X: X_batch.reshape([-1, self.img_dim, self.img_dim, 1])
                                             , self.y: y_batch
                                             , self.mask_with_labels: True})
                    print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(iteration
                                                                              , n_iterations_per_epoch
                                                                              , iteration * 100 / n_iterations_per_epoch
                                                                              , loss_train)
                                                                              , end="")

                # At the end of each epoch, measure the validation loss and accuracy:
                loss_vals = []
                acc_vals = []
                for iteration in range(1, n_iterations_validation + 1):
                    start = (iteration-1)*batch_size
                    stop = iteration*batch_size
                    X_batch, y_batch = self.X_scaled_val[start:stop], self.y_np_val[start:stop]   
                    loss_val, acc_val = sess.run([self.loss, self.accuracy]
                                                 , feed_dict={self.X: X_batch.reshape([-1, self.img_dim, self.img_dim, 1])
                                                 , self.y: y_batch})
                    loss_vals.append(loss_val)
                    acc_vals.append(acc_val)
                    print("\rEvaluating the model: {}/{} ({:.1f}%)".format(iteration
                                                                           , n_iterations_validation
                                                                           , iteration * 100 / n_iterations_validation)
                                                                           , end=" " * 10)
                loss_val = np.mean(loss_vals)
                acc_val = np.mean(acc_vals)
                print("\rEpoch: {}  Val accuracy: {:.4f}%  Loss: {:.6f}{}".format(epoch + 1
                                                                                  , acc_val * 100
                                                                                  , loss_val
                                                                                  , " (improved)" if loss_val < best_loss_val else ""))

                # And save the model if it improved:
                if loss_val < best_loss_val:
                    save_path = self.saver.save(sess, self.save_dir)
                    best_loss_val = loss_val


        # Evaluate Model Performance
        n_iterations_test = len(self.X_scaled_test)// batch_size
        with tf_.Session() as sess:
            self.saver.restore(sess, self.save_dir)

            loss_tests = []
            acc_tests = []
            for iteration in range(1, n_iterations_test + 1):
                start = (iteration-1)*batch_size
                stop = iteration*batch_size
                X_batch, y_batch = self.X_scaled_test[start:stop], self.y_np_test[start:stop]   
                loss_test, acc_test = sess.run([self.loss, self.accuracy]
                                               ,feed_dict={self.X: X_batch.reshape([-1, self.img_dim, self.img_dim, 1]), self.y: y_batch})                
                loss_tests.append(loss_test)
                acc_tests.append(acc_test)
                print("\rEvaluating the model: {}/{} ({:.1f}%)".format(iteration
                                                                       , n_iterations_test
                                                                       , iteration * 100 / n_iterations_test)
                                                                       , end=" " * 10)
            loss_test = np.mean(loss_tests)
            acc_test = np.mean(acc_tests)
            print("\rFinal test accuracy: {:.4f}%  Loss: {:.6f}".format(acc_test * 100, loss_test))



        batch_labels = []
        predictions = []
        n_iterations_test = len(self.X_scaled_test)// batch_size

        with tf_.Session() as sess:
            self.saver.restore(sess, self.save_dir)

            for iteration in range(1, n_iterations_test + 1):
                start = (iteration-1)*batch_size
                stop = iteration*batch_size
                X_batch, y_batch = self.X_scaled_test[start:stop], self.y_np_test[start:stop]   
                batch_labels.append(sess.run([self.y_pred]
                                             ,feed_dict={self.X: X_batch.reshape([-1, self.img_dim, self.img_dim, 1]), self.y: y_batch}))           

            len_labels = len(batch_labels)
            np_batch_labels = np.array(batch_labels) # convert to a numpy array
            np_2d_batch_labels = np_batch_labels[0:len_labels,0] # convert from 3d to 2d array
            predictions = np_2d_batch_labels.reshape(-1) # convert from a 2d to a 1d array

            len_pred = len(predictions)
            test_vals = self.y_np_test[0:len_pred]
            con_mat = confusion_matrix(test_vals, predictions)
            con_df = pd.DataFrame(data=con_mat, index=CATEGORIES, columns=CATEGORIES)

            print(con_df)
            print('\n')

            # accuracy: (tp + tn) / (p + n)
            accuracy = accuracy_score(test_vals, predictions)
            print('Accuracy: %f' % accuracy)
            # precision tp / (tp + fp)
            precision = precision_score(test_vals, predictions, average='weighted')
            print('Precision: %f' % precision)
            # recall: tp / (tp + fn)
            recall = recall_score(test_vals, predictions, average='weighted')
            print('Recall: %f' % recall)
            # f1: 2 tp / (2 tp + fp + fn)
            f1 = f1_score(test_vals, predictions, average='weighted')
            print('F1 score: %f' % f1)


        '''
        n_samples = 5
        sample_images = cap_X_test[:n_samples].reshape([-1, CAP_IMG_SIZE, CAP_IMG_SIZE, 1])
        with tf_.Session() as sess:
          saver.restore(sess, CAP_CHECKPT_LOG_DIR)
          caps2_output_value, decoder_output_value, y_pred_value = sess.run([caps2_output
                                                                            , decoder_output, y_pred]
                                                                            , feed_dict={X: sample_images, y: np.array([], dtype=np.int64)})

        sample_images = sample_images.reshape(-1, CAP_IMG_SIZE, CAP_IMG_SIZE)
        reconstructions = decoder_output_value.reshape([-1, CAP_IMG_SIZE, CAP_IMG_SIZE])

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
          plt.axis("off")
    
        plt.show()
        '''


    def create_loss_variable(self):

        alpha = 0.0005
        return tf_.add(self.margin_loss, alpha * self.reconstruction_loss, name="loss")
        
    def create_reconstruction_loss_variable(self):

        # The Decoder is a 3 Layer Dense Network ** Automate input sizes **
        n_hidden1 = 1024 # 512
        n_hidden2 = 2048 # 1024
        n_output = (self.img_dim)**2 

        with tf_.name_scope("decoder"):
            hidden1 = tf_.layers.dense(self.decoder_input, n_hidden1, activation=tf_.nn.relu, name="hidden1")
            hidden2 = tf_.layers.dense(hidden1, n_hidden2, activation=tf_.nn.relu, name="hidden2")
            decoder_output = tf_.layers.dense(hidden2, n_output, activation=tf_.nn.sigmoid, name="decoder_output")

        # Mean squared difference between the input values
        X_flat = tf_.reshape(self.X, [-1, n_output], name="X_flat")
        squared_difference = tf_.square(X_flat - decoder_output, name="squared_difference")

        return tf_.reduce_mean(squared_difference, name="reconstruction_loss")

    def create_decoder_input_variable(self):
       
        # Use the ground truths for training, otherwise use the prediction values (unsupervised learning)
        reconstruction_targets = tf_.cond(self.mask_with_labels  # condition
                                          , lambda: self.y       # if True
                                          , lambda: self.y_pred  # if False
                                          , name="reconstruction_targets")

        # One hot encode the reconstruction mask
        reconstruction_mask = tf_.one_hot(reconstruction_targets, depth=self.caps2_n_caps, name="reconstruction_mask")

        # Re-shape the reconstruction mask by increasing the rank in order to make the multiplication possible
        reconstruction_mask_reshaped = tf_.reshape(reconstruction_mask, [-1, 1, self.caps2_n_caps, 1, 1], name="reconstruction_mask_reshaped")
        
        # Create a variable to hold the masked output label predictions
        caps2_output_masked = tf_.multiply(self.caps2_output, reconstruction_mask_reshaped, name="caps2_output_masked")

        return tf_.reshape(caps2_output_masked, [-1, self.caps2_n_caps * self.caps2_n_dims], name="decoder_input")


    def compute_margin_loss(self):

        # Margin loss constants
        m_plus = 0.9
        m_minus = 0.1
        lambda_ = 0.5

        caps2_output_norm = find_norm(self.caps2_output, axis=-2, keepdims=True, name="caps2_output_norm")

        # To allow for multiple entities, a separate margin loss is computed for each capsule.
        present_error_raw = tf_.square(tf_.maximum(0., m_plus - caps2_output_norm), name="present_error_raw")
        present_error = tf_.reshape(present_error_raw, shape=(-1, self.caps2_n_caps), name="present_error") # 10

        # Downweighting the loss for absent entities stops the learning from shrinking activity vector lengths for all entities.
        absent_error_raw = tf_.square(tf_.maximum(0., caps2_output_norm - m_minus), name="absent_error_raw")
        absent_error = tf_.reshape(absent_error_raw, shape=(-1, self.caps2_n_caps), name="absent_error") # 10

        # Compute the margin loss value for each instance multiplied by each label
        L = tf_.add(self.T * present_error, lambda_ * (1.0 - self.T) * absent_error, name="L")

        return tf_.reduce_mean(tf_.reduce_sum(L, axis=1), name="margin_loss")

    def compute_predicted_class(self):

        # Compute the length of the output vectors (class probabilities)
        y_proba = find_norm(self.caps2_output, axis=-2, name="y_proba")
        y_proba_argmax = tf_.argmax(y_proba, axis=2, name="y_proba")

        return tf_.squeeze(y_proba_argmax, axis=[1,2], name="y_pred") # remove unnecessary dimensions

    def create_weighted_sum_round_2_variable(self):

        weighted_predictions_round_2 = tf_.multiply(self.routing_weights_round_2, self.caps2_predicted, name="weighted_predictions_round_2")
        return tf_.reduce_sum(weighted_predictions_round_2, axis=1, keepdims=True, name="weighted_sum_round_2")

    def create_round_2_routing_weights_variable(self):

        raw_weights_round_2 = tf_.add(self.raw_weights, self.agreement, name="raw_weights_round_2")
        return tf_.nn.softmax(raw_weights_round_2, axis=2, name="routing_weights_round_2")

    def create_round_1_RBA_variable(self):

        caps2_output_round_1_tiled = tf_.tile(self.caps2_output_round_1, [1, self.caps1_n_caps, 1, 1, 1], name="caps2_output_round_1_tiled") # re-shape so there is one multiplication operation per label
        return tf_.matmul(self.caps2_predicted, caps2_output_round_1_tiled, transpose_a=True, name="agreement")

    def create_weighted_sum_variable(self):

        # multiply the routing weights by the predicted capsules
        weighted_predictions = tf_.multiply(self.routing_weights, self.caps2_predicted, name="weighted_predictions") 
        return tf_.reduce_sum(weighted_predictions, axis=1, keepdims=True, name="weighted_sum")

    def create_caps1_output_variable(self):

        caps1_output_expanded = tf_.expand_dims(self.caps1_output, -1, name="caps1_output_expanded")
        caps1_output_tile = tf_.expand_dims(caps1_output_expanded, 2, name="caps1_output_tile") # increased rank
        
        return  tf_.tile(caps1_output_tile, [1, 1, self.caps2_n_caps, 1, 1], name="caps1_output_tiled") 

    def create_weight_matrices_variable(self):

        init_sigma = 0.1
        Weight_init = tf_.random_normal(
            shape=(1, self.caps1_n_caps, self.caps2_n_caps, self.caps2_n_dims, self.caps1_n_dims)
            , stddev=init_sigma, dtype=tf_.float32, name="Weight_init")
        W = tf_.Variable(Weight_init, name="Weights")

        return W

    def calc_caps1_caps(self):
        return np.int64((((self.img_dim - (2 * (self.kernel_size-1))) / 2)**2) * self.caps1_n_maps)

    def create_conv_layers(self):

        # Define the convolutional layers that will feed into the primary capsule layer
        conv1_params = {
          "filters": 256,
          "kernel_size": self.kernel_size, 
          "strides": 1,
          "padding": "valid",
          "activation": tf_.nn.relu,
        }
        c1 = tf_.layers.conv2d(self.X, name="conv1", **conv1_params)

        conv2_params = {
          "filters": self.caps1_n_maps * self.caps1_n_dims, # 256 convolutional filters
          "kernel_size": self.kernel_size, 
          "strides": 2,
          "padding": "valid",
          "activation": tf_.nn.relu
        }
        c2 = tf_.layers.conv2d(c1, name="conv2", **conv2_params)

        return c1, c2

    def compute_weights(self):

        # Set the training class weight proportions in case the data could not be balanced using the augmentation step
        class_weights = {}
        weight_classes = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(self.labels_data), y=self.labels_data)
        for i in range(len(weight_classes)):
            class_weights.update({i:weight_classes[i]})

        return class_weights


    def split_data(self):

        X_train, X_test, y_train, y_test = train_test_split(self.feature_data, self.labels_data, test_size = self.test_size, random_state = self.rand_state)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = self.val_size, random_state = self.rand_state)
 
        return X_train, X_test, X_val, y_train, y_test, y_val


    def create_processed_data(self):

        # Set the data type
        X_train_data = self.X_train.astype('float32') 
        X_test_data = self.X_test.astype('float32')
        X_val_data = self.X_val.astype('float32')

        sc_X_train = X_train_data / 255
        sc_X_test = X_test_data / 255
        sc_X_val = X_val_data / 255

        # Change label data type to numpy int
        y_np_train = np.asarray(self.y_train, dtype=np.int32)
        y_np_test = np.asarray(self.y_test, dtype=np.int32)
        y_np_val = np.asarray(self.y_val, dtype=np.int32)

        return sc_X_train, sc_X_test, sc_X_val, y_np_train, y_np_test, y_np_val


    def get_raw_features(self):
        return self.X_train, self.X_test, self.X_val

    def get_raw_labels(self):
        return self.y_train, self.y_test, self.y_val

    def get_weights(self):
        return self.class_weights

    def get_features(self):
        return self.X_scaled_train, self.X_scaled_test, self.X_scaled_val

    def get_labels(self):
        return self.y_np_train, self.y_np_test, self.y_np_val

