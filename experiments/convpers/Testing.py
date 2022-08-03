from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from keras import models
from keras import layers
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import regularizers
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from convpers import CPT
from convpers import Vectorize
from convpers import Filters
from sklearn.model_selection import train_test_split
import gtda

def test_knn(data,y,k=3,cv=3):
    """Tests a nearest-neighbor model on a classification task and prints results.

    Parameters
    ----------
    data : array
        the data array contains one feature vector per data point
    y : list
        a list of class labels
    k : integer
        the nearest-neighbor parameter 
    cv : integer
        the number of folds to use for cross-validation
    Returns
    -------
    
    """
    neigh = KNeighborsClassifier(n_neighbors=k)
    cv_results = cross_validate(neigh, data, y, cv=cv)
    print("kNN Results: ",cv_results['test_score'])
    print("Average kNN Result: ", np.average(cv_results['test_score']))

def test_boosted_tree(data,y,cv=3):
    """Tests a boosted tree on a classification task and prints results. 10 estimators are used.

    Parameters
    ----------
    data : array
        the data array contains one feature vector per data point
    y : list
        a list of class labels
    cv : integer
        the number of folds to use for cross-validation
    Returns
    -------
    
    """
    gb_model = GradientBoostingClassifier(n_estimators = 10,random_state = 0)
    cv_results = cross_validate(gb_model, data, y, cv=cv)
    print("Gradient Boosting Results: ", cv_results['test_score'])
    print("Average Gradient Boosting Result: ", np.average(cv_results['test_score']))

def test_NN(data,y):
    """Tests a neural network model on a classification task and prints results.
    The model has two hidden layers with 100 units, uses RELU activations on the hidden layers
    and softmax on the final layer, optimizer is adam. Trains for 50 epochs with batch size 4.
    10% of the data is used for computing validation loss and accuracy.

    Parameters
    ----------
    data : array
        the data array contains one feature vector per data point
    y : list
        a list of class labels
    Returns
    -------
    
    """
    enc = OneHotEncoder(sparse=False)
    labels = enc.fit_transform(y.reshape(-1,1))
    network = models.Sequential()
    network.add(layers.Dense(units=100,input_dim = np.shape(data)[1], activation=layers.LeakyReLU()))
    network.add(layers.Dense(units=100, activation=layers.LeakyReLU()))
    network.add(layers.Dense(units=len(np.unique(y)), activation='softmax'))
    network.compile(loss='categorical_crossentropy',optimizer='nadam',metrics='accuracy')
    history = network.fit(data, labels,validation_split = 0.1, epochs=100, batch_size=4,verbose=0)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Deep Learning Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    print("Final Validation Accuracy: ",history.history['val_accuracy'][-1] )
