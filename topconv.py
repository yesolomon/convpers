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

scaler = StandardScaler()
def rescale(collection):
    (A,m,n) = np.shape(collection)
    collection = np.reshape(collection,(A,m*n))
    collection = scaler.fit_transform(collection)
    return np.reshape(collection,(A,m,n))


def im_2_persim(collection,B=10):
    cubpers = gtda.homology.CubicalPersistence()
    persim  = gtda.diagrams.PersistenceImage(n_bins=B)
    diagrams = cubpers.fit_transform(collection)
    im = persim.fit_transform(diagrams)
    return np.reshape(im,(len(im),2*B*B))


def topfiltlearn(X,y,filters,pool=False):
    np.random.seed(0)
    enc = OneHotEncoder(sparse=False)
    labels = enc.fit_transform(y.reshape(-1,1))

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
            
    total_diagrams = np.concatenate(diagrams,axis = 1)
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
    network.add(layers.Dense(units=10, activation='softmax'))
    network.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics='accuracy')
    history = network.fit(total_diagrams, labels,validation_split = 0.1, epochs=50, batch_size=4,verbose=0)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Deep Learning Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()