# Copyright 2020 The ASReview Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from os.path import isfile
from os import remove 
import logging
import numpy as np
from tensorflow.python.keras.backend import reshape
from tensorflow.python.keras.layers.core import Reshape

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Conv1D
    #from tensorflow.keras.layers import Reshape
    from tensorflow.keras.layers import MaxPool1D
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.callbacks import EarlyStopping 
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
    from tensorflow.keras import optimizers

except ImportError:
    TF_AVAILABLE = False
else:
    TF_AVAILABLE = True
    try:
        tf.logging.set_verbosity(tf.logging.ERROR)
    except AttributeError:
        logging.getLogger("tensorflow").setLevel(logging.ERROR)

from sklearn.naive_bayes import MultinomialNB
import optuna

from asreview.models.classifiers.base import BaseTrainClassifier
from asreview.utils import _set_class_weight
from tensorflow.keras import backend


def _check_tensorflow():
    if not TF_AVAILABLE:
        raise ImportError(
            "Install tensorflow package (`pip install tensorflow`) to use"
            " 'EmbeddingIdf'.")


class CNNSwitch(BaseTrainClassifier):

    name = "CNNSwitch"

    def __init__(self,
                 learning_rate=0.01,
                 epochs=2,
                 batch_size=32,
                 shuffle=True,
                 class_weight=30.0):

        """Initialize the 2-layer neural network model."""
        super(CNNSwitch, self).__init__()


        print("""
    ________________________________________    
    ____________ CNN-NB Switcher ___________
    ________________________________________    
    _________________________,▄▓▓▓▓████▓▓▄__
    _______________________▄▓█▓▀        ╠▓█▓
    _____________________,▓█▓`   ,,,»≤░░▄▓██
    ____________________]▓█▓▓▓██████████▓▀▀_
    ________________,,µ▄▓███▀╙`_____________
    ___________▄▄▓▓█▓▓▓▓▓▓▓██▓▄_____________
    ________▄▓██▀╙└          ╙▀██▄__________
    ______▄▓█▀└                 ╙██▄________
    ____▄██▀                      ▓█▓_______
    ___▓█▓   ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄,   ╟█▓⌐_____
    __▓█▌   ╟███████████████████   '╟█▓_____
    _▓█▓    ╟███▄█▄███████▄█▄▀██▌   │▓█▓____
    ▐██     ╟███████████████████▌   ░╙██⌐___
    ╫█▌     ╙███████████████████    ░░▓█▌___
    ▓█▌                             ░░╫█▌___
    ▓█▌                           ,░░░▓█▌___
    ╟█▓                           ░░░:██⌐___
    _██▌                         ░░░░▓█▓____
    _╙██╕                      ;░░░░▓██`____
    __╙██▄                   ,░░░░]▓█▓`_____
    ___└▓█▓,               ;░░░░░▄▓█▀_______
    _____╙▓█▓▄         ,≤░░░░░;▄▓█▀└________
    _______└▀▓█▓▓▄╦;░░░░░;:▄▓▓██▀¬__________
    ______,▄▓▓█▓▓▓████▓▓███▓▓▓█▓▓▄__________
    _____å█▀└       ╙██▓▀'      ╚▀█▓________
    _____╟█▄     ,≤░▄▓██         ▓█▓________
    ______╙▀█▓▓▓▓▓▓█▓▀.╙▓▓▓▓▓▓▓▓█▓▀_________
     
        """)
        self.learning_rate=learning_rate
        self.epochs=int(epochs)
        self.batch_size=int(batch_size)
        self.shuffle=shuffle
        
        self.class_weight=class_weight
        self._model=None
        self.verbose=0
        self.iteration=1
        self.input_dim=None
        self.switched = False

        self.delta=0.03
        self.patience=20
        self.n_trials = 50
        self.switchpoint = 750


    def fit(self, X, y):

        self.X = X
        self.y = y    

        if self.iteration < self.switchpoint:
            self._model = MultinomialNB(alpha=3.822)
            self._model.fit(X, y)
            
            print("Model: Naive Bayes. Iteration: ", self.iteration)           
            self.iteration = self.iteration+1  

        elif self.iteration >= self.switchpoint:
            self.switched = True

            if self.iteration == self.switchpoint:
                print("--------- SWITCHING MODELS ---------")

            if self._model is None or X.shape[1] != self.input_dim:
                self.input_dim = X.shape[1]
                
            if self.iteration == self.switchpoint or (self.iteration%150) == 0:          

                self.hpotrial = 1
                self.parameters = self.hpo()

            keras_model = _create_network(
                input_dim = self.input_dim, 
                nlayers=self.parameters["nlayers"],
                nfilters1=self.parameters["nfilters1"],      
                nfilters2=self.parameters["nfilters2"],      
                nfilters3=self.parameters["nfilters3"],      
                nfilters4=self.parameters["nfilters4"],      
                nfilters5=self.parameters["nfilters5"],     
                learning_rate = self.learning_rate, 
                verbose = self.verbose)

            self._model = KerasClassifier(keras_model, verbose=self.verbose)
            self.earlystop = EarlyStopping(monitor='loss', mode='min', min_delta = self.delta, patience = self.patience, restore_best_weights= False)

            history = self._model.fit(
                self._add_dim(X),
                y,
                batch_size=self.batch_size,
                epochs=self.epochs,
                shuffle=self.shuffle,
                verbose=self.verbose,
                callbacks=[self.earlystop],
                class_weight=_set_class_weight(self.class_weight))

            print("Iteration: ", self.iteration, "Amount of epochs: ",len(history.history["loss"]))           
            self.iteration = self.iteration+1


    def hpo(self):
        def objective(trial):
    
            #HPO function
            nlayers = trial.suggest_int("nlayers",2,5)
            nfilters1 = trial.suggest_int("nfilters1",2,100)
            nfilters2 = trial.suggest_int("nfilters2",nfilters1,100)
            nfilters3 = trial.suggest_int("nfilters3",nfilters2,100)
            nfilters4 = trial.suggest_int("nfilters4",nfilters3,200)
            nfilters5 = trial.suggest_int("nfilters5",nfilters4,200)
        
            self.input_dim = self.X.shape[1]

            keras_model = _create_network(
                input_dim = self.input_dim, 
                nlayers=nlayers,
                nfilters1=nfilters1,      
                nfilters2=nfilters2,      
                nfilters3=nfilters3,      
                nfilters4=nfilters4,      
                nfilters5=nfilters5,     
                learning_rate = self.learning_rate, 
                verbose = self.verbose)

            self._model = KerasClassifier(keras_model, verbose=self.verbose)
            self.earlystop = EarlyStopping(monitor='loss', mode='min', min_delta = self.delta, patience = self.patience, restore_best_weights= True)

            history = self._model.fit(
                self._add_dim(self.X),
                self.y,
                batch_size=self.batch_size,
                epochs=self.epochs,
                shuffle=self.shuffle,
                verbose=self.verbose,
                callbacks=[self.earlystop],
                class_weight=_set_class_weight(self.class_weight))

            nepoch = len(history.history["loss"])-1
            print("Hpo trial: ", self.hpotrial,"/",self.n_trials)
            
            self.hpotrial = self.hpotrial+1

            return history.history["loss"][nepoch]
        
        print("---------------------------------------------------")
        print("STARTING HYPERPARAMETER OPTIMISATION NEURAL NETWORK")
        print("---------------------------------------------------")

        optuna.logging.set_verbosity(0)
        study = optuna.create_study()
        study.optimize(objective, n_trials = self.n_trials)

        self.parameters = study.best_params
        print("FOUND HYPERPARAMETERS: ", self.parameters)

        return self.parameters

    def _add_dim(self,X):
        if self.switched == True:           
            X = X.toarray()
            X = X.reshape((X.shape[0],X.shape[1],1))  
        return X

    def predict_proba(self, X):
        if self.switched == False:
            foo = self._model.predict_proba((X))
        elif self.switched == True:
            foo = self._model.predict_proba(self._add_dim(X))
        return foo

def _create_network(
                        input_dim,
                        nlayers,
                        nfilters1,      
                        nfilters2,      
                        nfilters3,      
                        nfilters4,      
                        nfilters5,      
                        learning_rate=0.01,
                        verbose=0):


    def model_wrapper():
        
        backend.clear_session()

        model = Sequential()

        if nlayers == 2:
            
            #Block 1
            model.add(Conv1D(input_shape = (input_dim, 1), filters = nfilters1, kernel_size = 2, activation='relu'))
            model.add(MaxPool1D(pool_size = 3))
            model.add(Dropout(0.2))
            #Block 2       
            model.add(Conv1D(filters = nfilters2, kernel_size = 2, activation='relu'))
            model.add(MaxPool1D(pool_size = 3))
            model.add(Dropout(0.2))
        
        if nlayers == 3:
            
            #Block 1
            model.add(Conv1D(input_shape = (input_dim, 1), filters = nfilters1, kernel_size = 2, activation='relu'))
            model.add(MaxPool1D(pool_size = 2))
            model.add(Dropout(0.2))
            #Block 2       
            model.add(Conv1D(filters = nfilters2, kernel_size = 2, activation='relu'))
            model.add(MaxPool1D(pool_size = 2))
            model.add(Dropout(0.2))
            #Block 3
            model.add(Conv1D(filters = nfilters3, kernel_size = 2, activation='relu'))
            model.add(MaxPool1D(pool_size = 2))
            model.add(Dropout(0.2))        
        
        if nlayers == 4:
            
            #Block 1
            model.add(Conv1D(input_shape = (input_dim, 1), filters = nfilters1, kernel_size = 2, activation='relu'))
            model.add(MaxPool1D(pool_size = 2))
            model.add(Dropout(0.2))
            #Block 2       
            model.add(Conv1D(filters = nfilters2, kernel_size = 2, activation='relu'))
            model.add(MaxPool1D(pool_size = 2))
            model.add(Dropout(0.2))
            #Block 3
            model.add(Conv1D(filters = nfilters3, kernel_size = 2, activation='relu'))
            model.add(MaxPool1D(pool_size = 2))
            model.add(Dropout(0.2))
            #Block 4      
            model.add(Conv1D(filters = nfilters4, kernel_size = 2, activation='relu'))
            model.add(MaxPool1D(pool_size = 2))
            model.add(Dropout(0.2))        

        if nlayers == 5:
            
            #Block 1
            model.add(Conv1D(input_shape = (input_dim, 1), filters = nfilters1, kernel_size = 2, activation='relu'))
            model.add(MaxPool1D(pool_size = 2))
            model.add(Dropout(0.2))
            #Block 2       
            model.add(Conv1D(filters = nfilters2, kernel_size = 2, activation='relu'))
            model.add(MaxPool1D(pool_size = 2))
            model.add(Dropout(0.2))
            #Block 3
            model.add(Conv1D(filters = nfilters3, kernel_size = 2, activation='relu'))
            model.add(MaxPool1D(pool_size = 2))
            model.add(Dropout(0.2))
            
            #Block 4      
            model.add(Conv1D(filters = nfilters4, kernel_size = 2, activation='relu'))
            model.add(MaxPool1D(pool_size = 2))
            model.add(Dropout(0.2))       
            #Block 5     
            model.add(Conv1D(filters = nfilters5, kernel_size = 2, activation='relu'))
            model.add(MaxPool1D(pool_size = 2))
            model.add(Dropout(0.2))      
        
        #Block Prediction
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.4))        
        
        model.add(Dense(24, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        # Compile model
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.RMSprop(learning_rate=learning_rate),
            metrics=['acc'])

        if verbose >= 1:
            model.summary()
    
        return model
    return model_wrapper
