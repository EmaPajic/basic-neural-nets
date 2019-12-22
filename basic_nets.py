#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: EmaPajic
"""

import scipy.io
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras import regularizers
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from keras import backend as K
from keras.callbacks import Callback

def read_data(file):
    d = scipy.io.loadmat(file)
    data = d['data']
    return data

def visualize_classes(data):
    class1 = []
    class2 = []
    for i in range(0, len(data[:, 2])):
        if data[i, 2] == 0:
            class1.append(data[i, 0:2])
        else:
            class2.append(data[i, 0:2])
    print(class1)
    plt.figure('Class 0')
    plt.scatter(*zip(*class1))
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.title('Class 0')
    plt.show()
    plt.figure('Class 1')
    plt.scatter(*zip(*class2))
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.title('Class 1')
    plt.show()
    
def split_training_test(x_data, y_data, test_size):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                        test_size = test_size)
    return x_train, x_test, y_train, y_test

def plot_history(history):
    plt.figure('Accuracy')
    plt.plot(history.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()
    plt.figure('Loss')
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

def plot_decision_borders(model):
    border_X = []
    border_Y = []
    border_input = []
    for x in np.linspace(-8, 8, 100):
        for y in np.linspace(-5, 8, 100):
            point = [x, y]
            border_input.append(point)
    
    border_input = np.array(border_input)
    predictions = model.predict(border_input)
    for i in range(0, len(predictions)):
        if(abs(predictions[i] - 0.5) <= 0.1):
            border_X.append(border_input[i][0])
            border_Y.append(border_input[i][1])
    
    X1 = []
    Y1 = []
    X2 = []
    Y2 = []
    for it in range(0, len(data)):
        if(data[it][2] == 0):
            X1.append(data[it][0])
            Y1.append(data[it][1])
        else:
            X2.append(data[it][0])
            Y2.append(data[it][1])
    
    plt.figure('Border')
    plt.title('Decision border')
    plt.scatter(X1, Y1, marker = '3', label = 'Class 0')
    plt.scatter(X2, Y2, marker = '4', label = 'Class 1')
    plt.gca().legend(('Class 0', 'Class 1'))
    plt.scatter(border_X, border_Y, marker = '.', color = 'black')


def plot_confusion_matrix(predictions, y, train_test):
    conf_mat = confusion_matrix(y, predictions)
    c_mat = [[0,0],[0,0]]
    c_mat[0][0] = conf_mat[0][0] / (conf_mat[0][0] + conf_mat[0][1])
    c_mat[0][1] = conf_mat[0][1] / (conf_mat[0][0] + conf_mat[0][1])
    c_mat[1][0] = conf_mat[1][0] / (conf_mat[1][0] + conf_mat[1][1])
    c_mat[1][1] = conf_mat[1][1] / (conf_mat[1][0] + conf_mat[1][1])
    df_cm = pd.DataFrame(c_mat, index = ["Class 0", "Class 1"],
                  columns = ["Class 0", "Class 1"])
    plt.figure('Confusion matrix on ' + train_test)
    sn.heatmap(df_cm, annot=True, cmap="YlGnBu")
    plt.title('Confusion matrix on ' + train_test)
    plt.show()
    
    
def neural_net(layer_dim, transfer_functions, regularization):
    model = Sequential()
    model.add(Dense(layer_dim[0], input_dim=2, activation=transfer_functions[0],
                    kernel_regularizer = regularizers.l2(regularization)))
    for i in range(1, len(layer_dim)):
        model.add(Dense(layer_dim[i], activation=transfer_functions[i],
                        kernel_regularizer = regularizers.l2(regularization)))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model

def train_neural_net(model, x, y, best_epoch, class_w):
    history = model.fit(x, y, epochs = best_epoch, batch_size = 1000, 
                        class_weight = {0: class_w,
                                        1: 1})
    return history
   
def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


class my_metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
 
    def on_epoch_end(self, epoch, logs={}):
        val_predict = K.constant((np.asarray(
                self.model.predict(self.validation_data[0]))).round())
        val_targ = K.constant(self.validation_data[1])
        _val_f1 = f1_m(val_targ, val_predict)
        _val_recall = recall_m(val_targ, val_predict)
        _val_precision = precision_m(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        return 

def find_best_net(x, y, net_layer_dims, net_transfer_functions,
                  net_regularizations, class_weights):
    
    best_layer_dim = net_layer_dims[0]
    best_transfer_functions = net_transfer_functions[0]
    best_regularization = net_regularizations[0]
    best_f1 = 0
    best_class_w = class_weights[0]
    best_epoch = 0
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.2)
    for i in range(0, len(net_layer_dims)):
        layer_dim = net_layer_dims[i]
        transfer_functions = net_transfer_functions[i]
        for regularization in net_regularizations:
            for class_w in class_weights:
                metrics = my_metrics()
                model = neural_net(layer_dim, transfer_functions,
                                   regularization) 
                
                history = model.fit(x, y, epochs = 500, batch_size = 1000, 
                        class_weight = {0: class_w,
                                        1: 1},
                        validation_data = (x_val, y_val),
                        callbacks = [metrics])
                val_f1 = metrics.val_f1s
                curr_best_epoch = K.argmax(val_f1).eval(session=K.get_session())
                f1 = val_f1[curr_best_epoch].eval(session=K.get_session())
                if f1 > best_f1:
                    best_f1 = f1
                    best_layer_dim = layer_dim
                    best_transfer_functions = transfer_functions
                    best_regularization = regularization
                    best_epoch = curr_best_epoch;
                    best_class_w = class_w
                    
    model = neural_net(best_layer_dim, best_transfer_functions,
                               best_regularization) 
    return model, best_epoch, best_class_w       

def test_neural_net(model, x, y):
    predictions = model.predict(x)
    rounded = [round(x[0]) for x in predictions]
    return rounded
    
def stats(y, rounded, train_test):
    print(train_test)
    target_names = ['class 0', 'class 1']
    print(classification_report(y, rounded, target_names = target_names))
    
if __name__ == '__main__':
    data = read_data('data.mat')
    #visualize_classes(data)
    x_train, x_test, y_train, y_test = split_training_test(data[:, 0:2],
                                                           data[:, 2], 0.2)
    #try different nets
    net_layer_dims = [[7, 6, 4, 1], [7, 6, 4, 1], [7, 6, 4, 1], [2, 2, 2, 2, 1],
                      [2, 2, 2, 2, 1], [2, 2, 2, 2, 1],
                      [5, 10, 60, 60, 60, 10, 5, 1],
                      [5, 10, 60, 60, 60, 10, 5, 1],
                      [5, 10, 60, 60, 60, 10, 5, 1]]
    net_transfer_functions = [['relu', 'relu', 'relu', 'sigmoid'],
                              ['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid'],
                              ['tanh', 'tanh', 'tanh', 'tanh'],
                            ['tanh', 'tanh', 'tanh', 'tanh', 'sigmoid'],
                            ['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid',
                             'sigmoid'],
                            ['relu', 'relu', 'relu', 'relu', 'sigmoid'],
                            ['tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'tanh',
                             'tanh', 'tanh'], 
                            ['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid',
                             'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid'],
                            ['relu', 'relu', 'relu', 'relu', 'relu', 'relu',
                             'relu', 'sigmoid']]
    net_regularizations = [0.01, 0.1, 0.4, 0.7]
    class_weights = [2, 3, 4, 5]
    best_epoch = 100
    best_class_w = 2
    nn, best_epoch, best_class_w = find_best_net(x_train, y_train,
                                                 net_layer_dims,
                                                 net_transfer_functions,
                                                 net_regularizations,
                                                 class_weights)
    
    #best net
    history = train_neural_net(nn, x_train, y_train, best_epoch, best_class_w)
    plot_history(history)
    plot_decision_borders(nn)
    train_set_predictions = test_neural_net(nn, x_train, y_train)
    stats(y_train, train_set_predictions, 'train set')
    plot_confusion_matrix(train_set_predictions, y_train, 'train set')
    test_set_predictions = test_neural_net(nn, x_test, y_test)
    stats(y_test, test_set_predictions, 'test set')
    plot_confusion_matrix(test_set_predictions, y_test, 'test set')
