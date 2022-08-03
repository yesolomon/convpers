import numpy as np
import gtda.homology
import gtda.diagrams
from scipy.signal import convolve2d


def im_to_pers(collection):
    """Computes persistence diagrams in dimension 0 and 1 for a collection of 2D images

    Parameters
    ----------
    collection : list of numpy arrays
        The list of images

    Returns
    -------
    diagrams : a list of persistence diagram pairs (dimension 0 and 1)
    """
    cubpers = gtda.homology.CubicalPersistence()
    diagrams = cubpers.fit_transform(collection)
    return diagrams



def CPT(X,filters):
    """Computes the Convolution Persistence Transform for a collection of filters

    Parameters
    ----------
    X : list of numpy arrays
        The list of images
    filters: list of numpy arrays
        The list of filters

    Returns
    -------
    diagrams: a list of lists of persistence diagram pairs (dimensions 0 and 1)
        The outer list corrsponds to different filters, an the inner lists to different images.
        If there are M filters and N images, then we return a list of length M, each element of which
        is a list of length N, each element of which is a pair of persistence diagrams
    """    
    datasets = []
    #For each filter...
    for f in filters:
        #Compute all convolutions...
        Xf = []
        for im in X:
            Xf.append(convolve2d(im,f,'same'))
        #and add those convolutions, as a list, to the collection of datasets
        datasets.append(Xf)

    diagrams = []
    #For each dataset (i.e. filter)...
    for D in datasets:
        #Compute the list of diagrams and at that to 'diagrams'
        diagrams.append(im_to_pers(D))

    return diagrams