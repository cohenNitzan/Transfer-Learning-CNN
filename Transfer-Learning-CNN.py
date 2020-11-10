import scipy
from scipy import io
import cv2
import numpy as np
import keras
from keras.models import load_model, Model
from keras import optimizers, Sequential, layers, regularizers, Sequential
from keras.layers import merge, Input, Dense, Activation, Flatten, Dropout, BatchNormalization, Concatenate, Conv2D
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.applications.resnet_v2 import ResNet50V2
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix, classification_report, \
    auc, roc_curve
from sklearn import preprocessing
import os
from os import listdir
from os.path import isfile, join
import pickle
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sn
import skimage
import copy
import pandas as pd
from pprint import pprint
import time
import random

def GetDefaultParameters():
    '''
    Set all the parameters of all the functions
    :return: A dict containing the default experiment parameters
    '''

    DataPath = "C:/Users/user/PycharmProjects/task2/FlowerData"
    test_images_indices = list(range(301, 473))

    Tune = {
        'tune': False,
        'base': False,
        'Aug': True
    }

    Data = {
        'data_path': DataPath,
        'image_size': (224, 224),
        'test_images_indices': test_images_indices,
        'train_size': 472 - len(test_images_indices),
        'valid_set': 0.2
    }

    NetworkParams = {
        'dropout': 0.5,
        'regularizer': 0.05,
        'FC_neurons': 2000,
        'N_layers': 10,
        'lr': 0.005
    }

    TuningTrain = {
        'dropout': [0.5, 0.2, 0],
        'regularizer': [0.05, 0.01, 0],
        'FC_neurons': [500, 1000, 2000],
    }

    TrainModel = {
        'epochs': 3,
        'batch_size': 20,
        'pathModel': 'improved_model.hdf5'
    }

    Augmantation = {
        'rotation_range': 30,
        'width_shift_range': 0.2,
        'height_shift_range': 0.2,
        'shear_range': 30,
        'zoom_range': 0.2,
        'horizontal_flip': True,
        'vertical_flip': True,
        'data_format': 'channels_last',
        'aug_real_ratio': 10
    }


    return {
        'Tune': Tune,
        'Data': Data,
        'NetworkParams': NetworkParams,
        'TuningTrain': TuningTrain,
        'TrainModel': TrainModel,
        'Augmantation': Augmantation,

    }


def GetData(params_data):
    '''
    gets the data
    :param params_data: the relevant parameters.
    :return: A dict:
                dict Data which contains:
                    Data (ndarray, [num_of_imagesX224X224X3])
                    Labels(vector, [1Xnum_of_images])
    '''

    raw_data = scipy.io.loadmat(params_data['data_path'] + '/FlowerDataLabels.mat')
    images = raw_data['Data'][0, :]
    labels = raw_data['Labels']
    resized_images = []
    for i in images:
        i = (i / 127.5) - 1 # normalization
        resized = cv2.resize(i, params_data['image_size'])
        resized_images.append(resized)
    images = np.asarray(resized_images)
    return {'Data': images,
            'Labels': labels
            }


def SplitData(data, labels, params_data, valid):
    '''
    Splits the data and labels according to the required indices defined in params.
    :param data: dict containing the data for splitting
    :param labels: labels for splitting
    :param params_data: dict containing all the data
    :param valid: boolean flag- if true splitting for validation
    :return:
            1. A dict containing train data and labels
            2. A dict containing validation/test data and labels
    '''

    if labels.shape[0] == 1:  # reduce dimension
        labels = labels[0, :]

    if valid:
        valid_size = int(np.floor((params_data['train_size']) * (params_data['valid_set'])))
        val_range = np.random.choice(params_data['train_size'], size=valid_size, replace=False)
        train_range = list(set(list(range(0, params_data['train_size']))) - set(val_range))
        test_data = data[val_range, :, :, :]
        train_data = data[train_range, :, :, :]
        test_labels = labels[val_range]
        train_labels = labels[train_range]

    else:
        test_range = list(np.asarray(params_data['test_images_indices']) - 1)
        train_range = list(set(list(range(0, 472))) - set(test_range))
        test_data = data[test_range, :, :, :]
        train_data = data[train_range, :, :, :]
        test_labels = labels[test_range]
        train_labels = labels[train_range]

    return {
        'train_data': train_data,
        'train_labels': train_labels,
        'test_data': test_data,
        'test_labels': test_labels
    }


def base_Network():
    '''
    Build task transfer from ResNet50V2 - remove last layer and add layer dense sigmoid
    :return the new base model
    '''
    # Create the base model from the pre-trained model ResNet50V2
    base_model = ResNet50V2(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='avg')
    #base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    # Freeze the convolutional base
    outputX = base_model.layers[-1].output
    out = Dense(1, activation='sigmoid')(outputX)
    base_model = Model(base_model.input, outputs=out)
    for layer in base_model.layers[:-1]:
        layer.trainable = False
    #base_model.summary()
    optimizer = optimizers.RMSprop(lr=0.005)
    base_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return base_model


def N_Last_Layers_Network(NetworkParams):
    '''
    Build task transfer from ResNet50V2 - remove  N last layers and add layer dense sigmoid
    :return the new base model
    '''
    # Create the base model from the pre-trained model ResNet50V2
    model = ResNet50V2(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='avg')
    model.summary()
    outputX = model.layers[-1].output
    out = Dense(1, activation='sigmoid')(outputX)
    Improved_model = Model(model.input, outputs=out)
    # Freeze the convolutional base
    for layer in Improved_model.layers[:-NetworkParams['N_layers']]:
        layer.trainable = False
    Improved_model.summary()

    optimizer = optimizers.Adam(lr=0.005)
    Improved_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return Improved_model


def N_last_layers_AND_more_Network(NetworkParams):
    '''
    Build task transfer from ResNet50V2 - remove N last layers and add dropout, FC layer and layer dense sigmoid.
    :return the new base model
    '''
    # Create the base model from the pre-trained model ResNet50V2
    model = ResNet50V2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    #model.summary()
    # Freeze the convolutional base
    for layer in model.layers[:-NetworkParams['N_layers']]:
        layer.trainable = False
    global_average_layer = Flatten(name='GAvgPool2D')
    drop1 = Dropout(rate=NetworkParams['dropout'])
    fullyConnected1 = Dense(1000, activation='relu', name='fullyConnected1', kernel_regularizer=regularizers.l2(0.05))
    drop2 = Dropout(rate=NetworkParams['dropout'])
    fullyConnected2 = Dense(1000, activation='relu', name='fullyConnected2', kernel_regularizer=regularizers.l2(0.05))
    drop3 = Dropout(rate=NetworkParams['dropout'])
    prediction_layer = Dense(1, activation='sigmoid', name='predictions')
    #drop2, fullyConnected2, drop3,
    Improved_model = Sequential(
        [model, drop1, global_average_layer, fullyConnected1, prediction_layer])
    #Improved_model.summary()

    optimizer = optimizers.Adam(lr=0.005)
    Improved_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return Improved_model

def TuneHyperParameters_improveNet(splitValid, hp_params, train_params, aug_params, net_params):
    '''
    This function find the best hyper-parameters
    :param splitValid: the training data, training labels, validation data, validation labels
    :param hp_params: dictionary containing the parameters to tune
    :param train_params: dictionary containing the training parameters
    :param aug_params: dictionary containing the parameters for data augmentation
    :param net_params: dictionary containing the network parameters
    :return: summary of the tuning process and the best hyper-parameter found in the process
    '''

    # build summary
    HP = list(hp_params.keys())
    columns = HP.copy()
    columns.extend(['accuracy', 'best amount of epochs'])
    summary = pd.DataFrame(columns=columns) # create new data frame with 0 rows and 5 colums (3 HP and 2 for results)

    # initialized
    best_accuracy = 0
    optimal_epochs = 0
    best_params = net_params.copy()
    delta_acc = 1
    df = pd.DataFrame(hp_params)
    while delta_acc > 0: # loop until no improvment in accuracy
        delta_acc = 0
        for hp in HP: #run on all 3 HP (dropout, regularizer, FC_neurons)
            params = best_params.copy()
            for val in hp_params[hp]:
                params[hp] = val
                improve_model = N_last_layers_AND_more_Network(params)
                data_gen = ImageDataGenerator(rotation_range=aug_params['rotation_range'], width_shift_range=aug_params['width_shift_range'],
                                              height_shift_range=aug_params['height_shift_range'], shear_range=aug_params['shear_range'],
                                              zoom_range=aug_params['zoom_range'],  horizontal_flip=aug_params['horizontal_flip'],
                                              vertical_flip=aug_params['vertical_flip'], data_format=aug_params['data_format'])
                train_generator = data_gen.flow(splitValid['train_data'], splitValid['train_labels'], batch_size=train_params['batch_size'])
                hist = improve_model.fit_generator(train_generator, validation_data=(splitValid['test_data'], splitValid['test_labels']),
                                           steps_per_epoch=len(splitValid['train_data']) // train_params['batch_size'], epochs=train_params['epochs'])
                acc, best_epochs = analyze_run(hist)
                # store summary
                param_config = [params[x] for x in HP] #enter row to summary table with current params
                param_config.extend([acc, best_epochs]) #extend row with results
                temp_summary = pd.DataFrame([param_config], columns=columns)
                summary = temp_summary if summary.size == 0 else summary.append(temp_summary, ignore_index=True)
                pprint(summary)

                # save new best hyper parameters
                if acc > best_accuracy:
                    delta_acc = acc - best_accuracy
                    best_params = params.copy()
                    best_accuracy = acc
                    optimal_epochs = best_epochs

    return summary, best_params, optimal_epochs

def analyze_run(hist):
    '''
    Analyze the run and returns the median in the top 5 accuracies and the weighted average num of best epochs
    :param hist: the log file of the run (History file)
    :return: median acc and optimal number of epochs
    '''
    all_acc = np.array(hist.history['val_accuracy'])
    epochs = np.argpartition(all_acc, -5)[-5:] #TODO change 2 to 5 #epoch start drom 0, change to 1, 2...
    acc = all_acc[epochs]
    median_acc = np.median(acc)
    weighted_best_epoch = np.average(epochs, weights=acc)
    return median_acc, weighted_best_epoch

def Train(params_trainModel, model, trainData, trainLabels, validData, validLabels):
    """
    train the model on the given data
    :param model: the chosen model to train
    :param trainData: the trainning images (Tensor, [num_of_images, 224, 224, 3])
    :param trainlabels: vector of training labels (vector, [num_of_images, 1])
    :param params_trainModel: a dictionary containing the training params
    :param validData: the validition images (Tensor, [num_of_images, 224, 224, 3])
    :param validLabels: vector of validition labels (vector, [num_of_images, 1])
    :return training score
    """
    t = time.time()
    # fit
    print('Training the network')
    # Save the model according to the conditions
    checkpoint = ModelCheckpoint(params_trainModel['pathModel'], monitor='val_accuracy', verbose=1, save_best_only=True,
                                 mode='max')
    # early stopping
    earlystopper = EarlyStopping(monitor='val_accuracy', patience=2, verbose=1)
    callbacks_list = [checkpoint, earlystopper]
    hist = model.fit(trainData, trainLabels, batch_size=params_trainModel['batch_size'],
                     epochs=params_trainModel['epochs'], verbose=1, callbacks=callbacks_list,
                     validation_data=(validData, validLabels), shuffle=True)
    print('Training time: %s' % (time.time() - t))
    (loss, accuracy) = model.evaluate(validData, validLabels)
    print("[Train_Info] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))
    # model.save(params_trainModel['pathModel'])
    return hist


def Train_Aug(params_trainModel, model, trainData, trainLabels, validData, validLabels, params_aug):
    '''
    train the model on the given data with augmantion
    '''
    t = time.time()
    data_gen = ImageDataGenerator(rotation_range=params_aug['rotation_range'],
                                  width_shift_range=params_aug['width_shift_range'],
                                  height_shift_range=params_aug['height_shift_range'],
                                  shear_range=params_aug['shear_range'],
                                  zoom_range=params_aug['zoom_range'], horizontal_flip=params_aug['horizontal_flip'],
                                  vertical_flip=params_aug['vertical_flip'], data_format=params_aug['data_format'])
    # fit
    # Save the model according to the conditions
    checkpoint = ModelCheckpoint(params_trainModel['pathModel'], monitor='val_accuracy', verbose=1, save_best_only=True,
                                 mode='max')
    # early stopping
    earlystopper = EarlyStopping(monitor='val_accuracy', patience=2, verbose=1)
    callbacks_list = [checkpoint, earlystopper]
    print('Training the network')
    train_generator = data_gen.flow(trainData, trainLabels, batch_size=params_trainModel['batch_size'])
    hist = model.fit_generator(train_generator, validation_data=(validData, validLabels),
                               steps_per_epoch=len(trainData) // params_trainModel['batch_size'],
                               epochs=params_trainModel['epochs'], callbacks=callbacks_list)
    print('Training time: %s' % (time.time() - t))
    (loss, accuracy) = model.evaluate(validData, validLabels, batch_size=params_trainModel['batch_size'], verbose=1)
    print("[Train_Info] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))
    #model.save(params_trainModel['pathModel'])
    return hist


def ploting(hist):
    '''
    plotinig train validation graph
    '''

    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    train_acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    xc = range(len(train_loss))

    plt.figure(1, figsize=(7, 5))
    plt.plot(xc, train_loss)
    plt.plot(xc, val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train', 'val'])
    # print plt.style.available # use bmh, classic,ggplot for big pictures
    plt.style.use(['classic'])

    plt.figure(2, figsize=(7, 5))
    plt.plot(xc, train_acc)
    plt.plot(xc, val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train', 'val'], loc=4)
    plt.style.use(['classic'])
    plt.show()


def print_Test_Result(model, testData, testLabels):
    '''
    recieves the pre-trained network, test data and labels and print the test loss, accuracy and error
    :param model: pre-trained network
    :param data: test data
    :return: predicted value based on the test data
    '''
    print('Testing the network')
    (loss, accuracy) = model.evaluate(testData, testLabels, batch_size=10, verbose=1)
    print(
        "[Test_Info] loss={:.4f}, accuracy: {:.4f}%, error: {:.4f}%".format(loss, accuracy * 100, (1 - accuracy) * 100))


def precision_recall(results, test_labels):
    '''
    computes the precision-recall plot and report the average precision
    :param results: the predicted score of each image in the test data computed by a trained network [num_of_images]
    :param test_labels: the ground true (binary)
    :return: the average precision value, precision, and recall
    '''
    # compute the averege precision
    average_precision = average_precision_score(test_labels, results)
    print('Average precision-recall score RF: {}'.format(average_precision))

    # compute the precision and the recall
    precision, recall, _ = precision_recall_curve(test_labels, results)

    # plot the curve
    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.show()
    return average_precision, precision, recall


def plotImages(predictedY_score, errors, path, test_indices):
    '''
    plot 5 worst images for each error type (1 and 2) and print for each image: 1)Error type, 2)CNN score, 3)Error index.
    :param predictedY_score: the predicted values for the images
    :param errors: errors on test data
    :param path: data path
    :param test_indices: the real index of the test set
    '''
    raw_data = scipy.io.loadmat(path + '/FlowerDataLabels.mat')
    all_images = raw_data['Data'][0, :]
    type1_errors_indices, type2_errors_indices = error_type_Sorting(predictedY_score, errors)
    # type 1
    if len(type1_errors_indices) == 0:
        print()
        print("There are no type 1 errors")
    else:
        five_worst_indices = calculate_largest_error(type1_errors_indices, errors)
        five_worst_indices_in_all_data = (np.array([test_indices]) - 1).flatten()[five_worst_indices]
        plot_5_images(1, all_images[five_worst_indices_in_all_data], predictedY_score[five_worst_indices])
    # type 2
    if len(type2_errors_indices) == 0:
        print()
        print("There are no type 2 errors")
    else:
        five_worst_indices = calculate_largest_error(type2_errors_indices, errors)
        five_worst_indices_in_all_data = (np.array([test_indices]) - 1).flatten()[five_worst_indices]
        plot_5_images(2, all_images[five_worst_indices_in_all_data], predictedY_score[five_worst_indices])


def error_type_Sorting(predictedY_score, errors):
    '''
    Separates errors types in to 2 lists containing the indexes of test set observations in which type 1 and type 2 errors were made
    :param predicted_y_score: the predicted CNN score of the images in the test set
    :param errors: all errors from test set
    :return: type1_errors_indices, and type2_errors_indices
    '''
    type1_errors_indices = []
    type2_errors_indices = []
    for i in range(0, len(errors)):
        if errors[i] > 0.5:  # if error (wrong classification)
            if predictedY_score[i] < 0.5:  # type 1 error: prediction=0 and ground truth is 1
                type1_errors_indices.append(i)
            else:  # type 2 error: prediction=1 and ground truth is 0
                type2_errors_indices.append(i)
    return type1_errors_indices, type2_errors_indices


def calculate_largest_error(errors_indices, errors):
    '''
    Find the 5 worst SPECIFIC error type (1 OR 2) indices on the test set
    :param errors_indices: all errors of SPECIFIC type (1 OR 2) from test set
    :param errors: all errors from test set
    :return: five worst errors indices (sort from biggest to smallest)
    '''
    errors_type = errors[errors_indices]
    ln = min(5, len(errors_indices))
    five_worst_indices_in_errors_type = errors_type.argsort()[-ln:][::-1].tolist()
    five_worst_indices = np.asarray(errors_indices)[five_worst_indices_in_errors_type]
    return five_worst_indices


def plot_5_images(type, images, predictedY_score):
    '''
    plot 5 worst images for SPECIFIC error type (1 OR 2)
    :param type: errors type
    :param images: 5 worst images of SPECIFIC type
    :param predictedY_score: the predicted values for the 5 worst images of SPECIFIC type
    '''
    if len(predictedY_score) < 5:
        print()
        print("There are only ", len(predictedY_score), " images of type ", type, " errors.")
    for i in range(0, len(predictedY_score)):
        print()
        print('Error type: ', type)
        print('CNN score: ', predictedY_score[i])
        print('Error index: ', i + 1)
        plt.imshow(images[i])
        plt.show()


#########----------------Main---------------#########
np.random.seed(1234)
random.seed(1234)
start = time.time()

# load data and prepare it
print("Loading and preparing the data ")
Params = GetDefaultParameters()
DandL = GetData(Params['Data'])
split = SplitData(DandL['Data'], DandL['Labels'], Params['Data'], False)
splitValid = SplitData(split['train_data'], split['train_labels'], Params['Data'], True)

# Tune
if Params['Tune']['tune']:
    HP =TuneHyperParameters_improveNet(splitValid, Params['TuningTrain'], Params['TrainModel'], Params['Augmantation'], Params['NetworkParams'])

# train the model and evaluate it on the test data
else:
    # create the network
    if Params['Tune']['base']:
        model = base_Network() #base model
    else:
        model = N_last_layers_AND_more_Network(Params['NetworkParams']) #improve model

    # train the data on network
    if Params['Tune']['Aug']:
        hist = Train_Aug(Params['TrainModel'], model, splitValid['train_data'], splitValid['train_labels'],
                         splitValid['test_data'], splitValid['test_labels'], Params['Augmantation'])
    else:
        hist = Train(Params['TrainModel'], model, splitValid['train_data'], splitValid['train_labels'],
                     splitValid['test_data'], splitValid['test_labels'])

    #ploting(hist)

    # Results
    best_model = load_model(Params['TrainModel']['pathModel'])
    predictedY_score = best_model.predict(split['test_data']).flatten()
    errors = abs(predictedY_score - split['test_labels'])

    # Report
    print_Test_Result(best_model, split['test_data'], split['test_labels'])
    print("running time = {}  seconds".format(int(time.time() - start)))
    precision_recall(predictedY_score, split['test_labels'])
    plotImages(predictedY_score, errors, Params['Data']['data_path'], Params['Data']['test_images_indices'])
