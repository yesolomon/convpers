import numpy as np
from sklearn.datasets import load_digits
import gtda.homology
import gtda.diagrams
import matplotlib.pyplot as plt
from  sklearn.preprocessing import StandardScaler
from skimage import filters
from scipy.signal import convolve2d
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from sklearn.ensemble import GradientBoostingClassifier
from keras import models
from keras import layers
from sklearn.preprocessing import OneHotEncoder
from skimage.measure import block_reduce
import time
from datetime import timedelta
from sklearn.decomposition import PCA 


scaler = StandardScaler()
def rescale(collection):
    (A,m,n) = np.shape(collection)
    collection = np.reshape(collection,(A,m*n))
    collection = scaler.fit_transform(collection)
    return np.reshape(collection,(A,m,n))

def im_2_pers(collection):
    cubpers = gtda.homology.CubicalPersistence()
    diagrams = cubpers.fit_transform(collection)
    return diagrams



def im_2_persim(collection,B=10):
    cubpers = gtda.homology.CubicalPersistence()
    persim  = gtda.diagrams.PersistenceImage(n_bins=B)
    diagrams = cubpers.fit_transform(collection)
    im = persim.fit_transform(diagrams)
    return np.reshape(im,(len(im),2*B*B))

def filters2diagrams(X,filters,pool=False):
    np.random.seed(0)
    
    X = rescale(X)

    
    if pool:
        X_pooling = []
        for im in X:
            X_pooling.append(block_reduce(im,(2,2),np.mean))
        X = X_pooling

    #start = time.time()
    datasets = [X]
    if filters:
        for f in filters:
            Xf = []
            for im in X:
                Xf.append(convolve2d(im,f,'same'))
            datasets.append(Xf)

 #   if pool:
 #       (a,b,c,d) = np.shape(datasets)
 #       pooled_datasets = np.zeros(shape = (a,b,int(c/2),int(d/2)))
 #       for id1, D in enumerate(datasets):
 #           for id2, im in enumerate(D):
 #               pooled_datasets[id1][id2] = block_reduce(im,(2,2),np.mean)
 #       datasets=pooled_datasets

    diagrams = []
    for D in datasets:
        diagrams.append(im_2_pers(D))

    return diagrams

def diagrams_2_images(collection,B=10):
    images = []
    persim  = gtda.diagrams.PersistenceImage(n_bins=B)
    for D in collection:
        im = persim.fit_transform(D)
        images.append(np.reshape(im,(len(im),2*B*B)))
    return images

def diagrams_2_tpers(collection):
    f = len(collection)
    s = len(collection[0])
    total_persistences = np.zeros((f,s,2))
    for i in range(f):
        for j in range(s):
            for (b,d,idx) in collection[i][j]:
                total_persistences[i,j,int(idx)]+=(d-b)**2
    return total_persistences




def concatenate_top_features(D):
    return np.concatenate(D,axis = 1)

def average_top_featres(D):
    return np.average(D,axis=0)   


def testmodels(data,y):
    enc = OneHotEncoder(sparse=False)
    labels = enc.fit_transform(y.reshape(-1,1))

    neigh = KNeighborsClassifier(n_neighbors=13)
    cv_results = cross_validate(neigh, data, y, cv=3)
    #print(sorted(cv_results.keys()))
    print("kNN Results: ",cv_results['test_score'])

    #Model 2: Boosted Trees.
    gb_model = GradientBoostingClassifier(n_estimators = 10,random_state = 0)
    cv_results = cross_validate(gb_model, data, y, cv=3)
    #print(sorted(cv_results.keys()))
    print("Gradient Boosting Results: ", cv_results['test_score'])

    #Model 3: Deep Learning
    network = models.Sequential()
    network.add(layers.Dense(units=100,input_dim = np.shape(data)[1], activation='relu'))
    network.add(layers.Dense(units=100, activation='relu'))
    network.add(layers.Dense(units=len(np.unique(y)), activation='softmax'))
    network.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics='accuracy')
    history = network.fit(data, labels,validation_split = 0.1, epochs=50, batch_size=4,verbose=0)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Deep Learning Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()



def topfiltlearn(X,y,filters,pool=False,average=False,proj=False):
    np.random.seed(0)
    enc = OneHotEncoder(sparse=False)
    labels = enc.fit_transform(y.reshape(-1,1))

    start = time.time()
    datasets = [X]
    if filters:
        for f in filters:
            Xf = []
            for im in X:
                Xf.append(convolve2d(im,f,'same'))
            datasets.append(Xf)

    for idx,D in enumerate(datasets):
        datasets[idx] = rescale(D)


    if pool:
        (a,b,c,d) = np.shape(datasets)
        pooled_datasets = np.zeros(shape = (a,b,int(c/2),int(d/2)))
        for id1, D in enumerate(datasets):
            for id2, im in enumerate(D):
                pooled_datasets[id1][id2] = block_reduce(im,(2,2),np.mean)
        datasets=pooled_datasets




    diagrams = []
    for D in datasets:
        diagrams.append(im_2_persim(D))
                
    if average:
        total_diagrams = np.average(diagrams,axis=0)     
    else:   
        total_diagrams = np.concatenate(diagrams,axis = 1)

    if proj:
        pca = PCA(0.9)
        total_diagrams = pca.fit_transform(total_diagrams)

    print("Shape of feature data: ", np.shape(total_diagrams))

    elapsed = (time.time() - start)
    print("Time to compute features: ", timedelta(seconds=elapsed))

    #print(np.shape(total_diagrams))

    #Model 1: KNN Classifier

    neigh = KNeighborsClassifier(n_neighbors=13)
    cv_results = cross_validate(neigh, total_diagrams, y, cv=3)
    #print(sorted(cv_results.keys()))
    print("kNN Results: ",cv_results['test_score'])

    #Model 2: Boosted Trees.
    gb_model = GradientBoostingClassifier(n_estimators = 10,random_state = 0)
    cv_results = cross_validate(gb_model, total_diagrams, y, cv=3)
    #print(sorted(cv_results.keys()))
    print("Gradient Boosting Results: ", cv_results['test_score'])

    #Model 3: Deep Learning
    network = models.Sequential()
    network.add(layers.Dense(units=100,input_dim = np.shape(total_diagrams)[1], activation='relu'))
    network.add(layers.Dense(units=100, activation='relu'))
    network.add(layers.Dense(units=len(np.unique(y)), activation='softmax'))
    network.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics='accuracy')
    history = network.fit(total_diagrams, labels,validation_split = 0.1, epochs=50, batch_size=4,verbose=0)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Deep Learning Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()