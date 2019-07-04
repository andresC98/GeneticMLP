#Genetic Algorithm Library (PonyGE2)
from fitness.base_ff_classes.base_ff import base_ff

#Math, Data Science and System libraries
import numpy  as np
import pandas as pd
import time , sys, os

# Scikit-earn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_digits
# Keras & tensorflow
import keras
from keras.layers import Dense 
from keras.models import Model, Sequential
from keras import optimizers, losses
from keras.utils import to_categorical
from keras import backend as K

import tensorflow as tf
import logging


class MLP(base_ff):
    maximise = True #we want to maximize our objective: model score.
    
    def __init__(self):
        tf.get_logger().setLevel(logging.ERROR)

        super().__init__()
        #class attributes (placeholders, later as init attributes)
        self.it         = 25
        self.lr         = 0.01
        self.verbose    = True
        self.optimizer  = optimizers.SGD(lr = self.lr, momentum = 0.9,nesterov = True)

        data = load_digits()
        X = data.data
        X = (X - np.min(X)) / (np.max(X) - np.min(X))
        y = data.target
        
        self.classes_, self.y = np.unique(y, return_inverse=True)
        self.n_classes = self.classes_.shape[0]
        #right now I wont be using testing split, so is very small
        self.X, self.X_test, self.y, self.y_test = train_test_split(X, y, test_size=0.1)

        self.X = np.array(self.X)
        self.X_test = np.array(self.X_test)
        self.y = to_categorical(self.y) 
        self.nn = None #Model (that will be tested, etc)

    def evaluate(self, ind, **kwargs):
        #print(ind.phenotype)
        inargs = {"xphe" : self.X.copy()}
        exec(ind.phenotype,inargs) #self.viu, msdX, self.nn initialized here
        #Obtain generated output from exec dictionary
        self.nn = inargs['nn']
        #print("Layers being used: ", self.nn.layers) #TODO: make print stm more verbose
        #Once GE has decided model, proceed to compile, test and evaluate it.
        self.nn.compile(loss = losses.categorical_crossentropy ,optimizer = self.optimizer, metrics=['accuracy'])
        self.nn.fit(x = self.X, y= self.y, validation_split=0.33,verbose=0,batch_size = 32, epochs = self.it)
                
        #Compute target variable to minimize.
	
	#score would be here
       
  
        return score #this will be the target to minimize by the GE algorithm.


