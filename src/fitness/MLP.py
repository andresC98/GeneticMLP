'''
Keras implementation of the Python's Genetic Algorithm approach
on PonyGE2 framework. 

This project is a Work in Progress, so right now it does not run.
'''


#Genetic Algorithm Library (PonyGE2)
from fitness.base_ff_classes.base_ff import base_ff

import numpy  as np
import pandas as pd
import pickle, time , sys, os
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

import keras
from keras.layers import Dense 
from keras.models import Model, Sequential
from keras import optimizers, losses
from keras.utils import to_categorical
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping

import tensorflow as tf
import logging

counter = 0

class MLP(base_ff):
    maximise = True #we want to maximize our objective: model score.
    global counter

    def __init__(self):
        tf.get_logger().setLevel(logging.ERROR)

        super().__init__()
        #class attributes (placeholders, later as init attributes)
        self.it         = 100
        self.lr         = 0.01
        self.verbose    = True
        self.optimizer  = optimizers.SGD(lr = self.lr, momentum = 0.9,nesterov = True)

        scaler = StandardScaler()

        data = load_digits()
        X = data.data
        y = data.target
        
        X, X_reserved, y, y_reserved  = train_test_split(X,y, random_state = 22,test_size = 0.25)
        

        self.classes_, self.y = np.unique(y, return_inverse=True)
        self.n_classes = self.classes_.shape[0]

        self.X, self.X_test, self.y, self.y_test = train_test_split(X, y, test_size=0.2)
        self.X = scaler.fit_transform(self.X)
        self.X_test = scaler.transform(self.X_test)
        
        #Keeping scaler mean and var for later use
        (s_mean, s_var) = scaler.mean_, scaler.var_
        self.fname = datetime.now().strftime('%H:%M:%S.csv')
        scaling_filename = "./results/"+self.fname[:8]+"/"+"scaling_factors.txt"
        os.makedirs(os.path.dirname(scaling_filename), exist_ok=True)
        with open(scaling_filename, 'wb') as f:
            pickle.dump((s_mean, s_var), f)

        self.X = np.array(self.X)
        self.X_test = np.array(self.X_test)
        self.y = to_categorical(self.y) 
        self.nn = None #Model (that will be tested, etc)

    def evaluate(self, ind, **kwargs):
        global counter
        try:
            inargs = {"xphe" : self.X.copy()}
            exec(ind.phenotype,inargs) 
            #Obtain generated output from exec dictionary
            self.nn = inargs['nn']
            #Once GE has decided model, proceed to compile, test and evaluate it.
            self.nn.compile(loss = losses.categorical_crossentropy ,optimizer = self.optimizer, metrics=['accuracy'])

            #Best network epoch resulted on training will be stored as individual
            fname = "./results/"+self.nsc_csv_name[:8]+"/networks/"+"Net"+str(counter)+'_fullmodel.hdf5'
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            early_stop = EarlyStopping(monitor='val_acc',mode='max', verbose=0,patience = 10)
            cp_save = ModelCheckpoint(fname, save_best_only=True,verbose=0, monitor='val_acc', mode='max')

            self.nn.fit(x = self.X, y= self.y, validation_split=0.33,verbose=0,batch_size = 32, epochs = self.it)
            
            #Best network resulting from training will be used for evaluation
            self.nn.load_weights(fname)
            #Compute target variable to minimize.
            model_loss, classification_ac = self.nn.evaluate(x = self.X_test, y = self.y_test, verbose = 0)
            
            #score would be here
            score = classification_acc 

            counter += 1
            K.clear_session()
        except:
            #In case of invalid individual, set objective to minimum
            score = 0.01
    
        return score #this will be the target to minimize by the GE algorithm.


