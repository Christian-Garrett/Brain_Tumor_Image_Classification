import os
import sys
from pathlib import Path
from collections import defaultdict

module_path = Path(__file__).parents[1]
sys.path.append(str(module_path))


class DataPipeline:

    """
    A class to handle data input, processing and evaluation for the convolutional
    and capsule neural network models which will be used in comparison to classify
    MRI brain scan images.

    ...

    Attributes
    ----------
    updated_image_dir : str
        The directory path that contains images after duplicate removal and
        augmentation balancing. If this parameter is passed in at instantiation
        these processes will be skipped.
    dirname : str
        The directory path containing the original unfiltered sample images
        for model training.
    hash_size : int
        Used to find duplicate images in the input folder that require deletion.
        Increasing the hash_size allows the Image algorithm from the Python Imaging
        Library (PIL) to store more detail in its hash, increasing its sensitivity
        to changes in detail
    initial_image_counts : dict
        Contains the number of unduplicated images in each input image folder
    image_pixel_data : list
        Each element contains the 2D image pixel data and the label encoded image
        category
    image_shape_counts : dict
        Each element contains the count of images with the same shape (dimensions)
        in the input data set
    image_shape_areas : dict
        Each element contains a count of images with the same shape area
    image_shape_dict : dict
        Each element contains a list with the image shape count and the corresponding
        area
    self.conv_resize : int
        The length and width dimension to resize the convolutional NN image input data
    self.cap_resize : int
        The length and width dimension to resize the capsule NN image input data
    target_label_dict : dict
        Each element contains the image folder name (tumor classification type) and
        the encoded integer label
    total_images : int
        The total number of input images combined across all classification labels
    label_data : list
        Classification label data that has been encoded for training after being saved 
        by the Torch library
    conv_data : list
        Image pixel data that will be used to train the convolutional model after all
        of the images have been resized to the same shape
    cap_data : list
        Image pixel data that will be used to train the capsule NN model after all of the 
        images have been resized to the same shape
    class_weights : dict
        Different weights assigned to each class to facilitate balance during training
    conv_scaler_map : dict
        Mean and variance info derived from applying the SciKit-Learn scaling
        operations on the data
    conv_X_train : list
        Processed training feature data input for the convolutional model
    conv_X_train_scaled : list
        Processed scaled training feature data for the convolutional model
    conv_X_test : list
        Processed test feature data input for the convolutional model
    conv_y_train : list
        Processed training label data input for the convolutional model
    conv_y_train_OHE : list
        Processed one hot encoded label training data for the convolutional model
    conv_y_test : list
        Processed test label data input for the convolutional model
    conv_y_test_OHE : list
        Processed one hot encoded label test data for the convolutional model
    best_conv_hyperparameters : dict
        Best performing model hyperparameters after tuning the convolutional model
    conv_model_callbacks : list
        Contains Keras model checkpoint and tensorboard objects used to track
        important metrics during convolutional model training
    conv_training_history : dict
        Contains all of the training history returned after fitting the model
    cap_features_path : str
        Directory path containing the processed image features for the capsule
        model
    cap_feature_data : list
        Capsule NN model training data that has been processed and saved using Torch
    cap_labels_path : str
        Directory path containing the processed image labels for the capsule NN model
    cap_label_data : list
        Capsule NN model label data that has been processed and saved using Torch
    cap_X_train : list
        Training feature processed data input for the capsule NN model
    cap_X_train_scaled : list
        Scaled training feature processed data input for the capsule model
    cap_y_train : list
        Processed training data input labels for the capsule model
    cap_X_test : list
        Processed test feature data input for the capsule NN model
    cap_X_test_scaled : str
        Processed scaled test feature data input for the capsule NN model
    cap_y_test : list
        Processed test data input labels for the capsule NN model
    cap_X_val : list
        Processed validation feature data input for the capsule NN model
    cap_X_val_scaled : str
        Scaled processed validation feature data input for the capsule NN model
    cap_y_val : str
        Processed validation label data input for the capsule NN model
    X_tensor : _
        Placeholder tensor for the input feature data
    y_tensor : _
        Placeholder tensor for the input label data
    y_tensor_onehot : _
        One hot encoding tensor for the input label data
    conv_layer1_tensor : _
        Layer tensor for the input layer of the convolutional model neural network
    conv_layer2_tensor : _
        Layer tensor for the second layer of the convolutional model neural network
    weights_tensor : _
        Variable tensor to hold weights used to balance training classes
    layer2_prediction_tensor : _
        Multiplication tensor to calculate capsule NN predictions using node output
        weights
    batch_size_tensor : _
        Shape tensor used to define the dimensions the tiled_weights_tensor
    routing_weights_round1_tensor : _
        Softmax tensor used to calculate the rounting by agreement (RBA) round
        one weights
    routing_weights_round1_sum_tensor : _
        Tensor used to calculate the sum of the round one routing weights
    routing_weights_round2_sum_tensor : _
        Tensor used to calculate the sum of the round two routing weights
    routing_agreement_tensor : _
        Round one matrix multiplication tensor to calculate routing by agreement 
        (RBA)
    layer2_round1_output_tensor : _
        Normalized vector holding the weighted sum of the first round of routing
        weights
    routing_weights_round2_tensor : _
        Softmax tensor used to calculate neuron activation for the convolutional
        output layer
    layer1_output_tensor : _
        Tensor to hold normalized and flattened layer one output data
    layer2_output_tensor : _
        Tensor to hold normalized and flattened layer two output data
    y_pred_tensor : _
        Values taken from the model output layer of the convolutional model and
        converted into target labels
    reconstruction_label_mask_tensor : _
        Tensor placeholder mask to support the reconstruction operation
    margin_loss_computation_tensor : _
        Tensor containing sum and mean values for combined marginal loss
    decoder_input_tensor : _
        Tensor input data for the decoder reconstruction operation
    reconstruction_loss_tensor : _
        Tensor containing decoder reconstruction loss values
    combined_margin_loss_tensor : _
        Combined present/absent entity loss
    combined_loss_tensor : _
        Tensor to hold the combined margin loss and reconstruction loss
    correct_tensor : _
        Equality tensor used to define the accuracy_tensor
    accuracy_tensor : _
        Tensor used to compare accuracy during model training
    tensor_training_optimizer : _
        Optimizer tensor used to determine accuracy and loss during model training
    initializer_object : _
        Running the initializer object sets the tensor attributes
    saver_object : _
        Object to save the output of the capsule NN model
    cap_test_prediction_list : list
        Predictions on the test set generated by the capsule NN model
    conv_test_prediction_list : list
        Predictions on the test set generated by the convolutional NN model


    Methods
    -------
    load_input_data(self):
        Load in the input image data and perform balancing and pruning if necesary. These
        data augmentation operations are only performed if an image directory containing
        the unprocessed images is not provided when the class is instantiated.
    perform_EDA(self):
        Obtain general info about the input data including category and shape counts, image 
        area info as well as pixel intensity metrics etc
    perform_data_preprocessing(self):
        Organizing the input data so that it is ready for manipulation
    perform_data_wrangling(self):
        Prepare the data so that it is ready for model input
    run_convolutional_model(self, model_name=None):
        Initialize, run and evaluate the convolutional NN model. Including an existing model
        name will avoid building a new model from scratch.
    run_capsule_model(self, model_folder=None):
        Initialize, run and evaluate the capsule NN model. Including an existing model
        name will avoid building a new model from scratch..        

    """

    from ML_Pipeline.Tools import (remove_duplicate_images,
                                   handle_imbalanced_input_data,
                                   balance_input_image_sets,
                                   augment_input_images,
                                   augment_image,
                                   load_processed_datasets,
                                   load_processed_dataset,
                                   create_weight_dict,
                                   scale_features,
                                   encode_labels,
                                   split_datasets,
                                   split_dataset,
                                   scale_convolutional_dataset,
                                   scale_capsule_dataset,
                                   print_classification_scores)

    from ML_Pipeline.EDA import (load_brain_scans,
                                 get_image_data,
                                 get_image_info,
                                 resize_images,
                                 shuffle_images,
                                 reshape_images,
                                 reshape_image,
                                 save_processed_images,
                                 save_processed_image)

    from ML_Pipeline.Convolutional_NN import evaluate_conv_model, get_model_metrics

    from ML_Pipeline.Capsule_NN import (init_cap_model,
                                        evaluate_cap_model,
                                        create_capsule_network_tensors,
                                        create_conv_layer_tensors,
                                        create_weight_tensor,
                                        create_layer1_output_tensor,
                                        create_routing_weights_round1_sum_tensor,
                                        create_RBA_round1_tensor,
                                        create_round2_routing_weights_tensor,
                                        create_routing_weights_round2_sum_tensor,
                                        compute_label_prediction_tensor,
                                        create_margin_loss_compute_tensor,
                                        create_decoder_reconstruction_input_tensor,
                                        create_reconstruction_loss_tensor,
                                        compute_combined_loss_tensor,
                                        create_prediction_tensors,
                                        create_RBA_computation_tensors,
                                        create_RBA1_tensors,
                                        create_RBA2_tensors,
                                        create_prediction_loss_tensors,
                                        create_reconstruction_decoder_tensors,
                                        create_utility_tensors,
                                        define_capsule_network_variables)

    def __init__(self, hash_size=250, conv_resize=256, cap_resize=56, color_channels=1, 
                 unfiltered_images_dir=None):

        self.updated_image_dir = os.path.join(module_path, "Input/Clean_Set_Updated/")
        self.dirname = \
            self.updated_image_dir if not unfiltered_images_dir else os.path.join(module_path,
                                                                                  f'Input/{unfiltered_images_dir}/')
        self.clean_images = True if unfiltered_images_dir else False
        self.hash_size = hash_size
        self.conv_resize = conv_resize
        self.cap_resize = cap_resize
        self.color_channels = color_channels
        self.total_images = 0
        self.initial_image_counts = dict()
        self.image_pixel_data = None
        self.image_shape_counts = defaultdict(int)
        self.image_shape_areas = dict()
        self.image_shape_dict = dict()
        self.target_label_dict = dict()
        self.label_data = None
        self.conv_data = None
        self.cap_data = None
        self.class_weights = None
        self.conv_scaler_map = None
        self.conv_X_train = None
        self.conv_X_train_scaled = None
        self.conv_X_test = None
        self.conv_y_train = None
        self.conv_y_train_OHE = None
        self.conv_y_test = None
        self.conv_y_test_OHE = None
        self.best_conv_hyperparameters = None
        self.conv_model_callbacks = None
        self.conv_training_history = None
        self.cap_features_path = None
        self.cap_feature_data = None
        self.cap_labels_path = None
        self.cap_label_data = None
        self.cap_X_train = None
        self.cap_X_train_scaled = None
        self.cap_y_train = None
        self.cap_X_test = None
        self.cap_X_test_scaled = None
        self.cap_y_test = None
        self.cap_X_val = None
        self.cap_X_val_scaled = None
        self.cap_y_val = None
        self.X_tensor = None
        self.y_tensor = None
        self.y_tensor_onehot = None
        self.conv_layer1_tensor = None
        self.conv_layer2_tensor = None
        self.weights_tensor = None
        self.layer2_prediction_tensor = None
        self.batch_size = None
        self.routing_weights_round1_tensor = None
        self.routing_weights_round1_sum_tensor = None
        self.routing_weights_round2_sum_tensor = None
        self.routing_agreement_tensor = None
        self.layer2_round1_output_tensor = None
        self.routing_weights_round2_tensor = None
        self.layer1_output_tensor = None
        self.layer2_output_tensor = None
        self.y_pred_tensor = None
        self.reconstruction_label_mask_tensor = None
        self.margin_loss_computation_tensor = None
        self.decoder_input_tensor = None
        self.reconstruction_loss_tensor = None
        self.combined_margin_loss_tensor = None
        self.correct_tensor = None
        self.accuracy_tensor = None
        self.tensor_training_optimizer = None
        self.initializer_object = None
        self.saver_object = None
        self.cap_test_prediction_list = None
        self.conv_test_prediction_list = None

    def load_input_data(self):

        if not self.updated_image_dir:

            self.remove_duplicate_images()
            self.handle_imbalanced_input_data()

        self.load_brain_scans()


    def perform_EDA(self):

        self.get_image_info()


    def perform_data_preprocessing(self):

        self.resize_images()
        self.shuffle_images()
        self.reshape_images()
        self.save_processed_images()


    def perform_data_wrangling(self):

        self.load_processed_datasets()
        self.create_weight_dict()
        self.split_datasets()
        self.scale_features()
        self.encode_labels()


    def run_convolutional_model(self, model_name=None):

        self.evaluate_conv_model() if not model_name else self.evaluate_conv_model(model_name)


    def run_capsule_model(self, model_folder=None):
        
        self.init_cap_model()
        self.evaluate_cap_model() if not model_folder else self.evaluate_cap_model(model_folder)
