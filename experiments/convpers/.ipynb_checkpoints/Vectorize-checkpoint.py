import numpy as np
import gtda.homology
import gtda.diagrams



def diagrams_to_images(collection,B=10):
    """Transform a list of lists of persistence diagrams (indexed by filter, image) into a list of lists
    of persistence image vectors, see https://jmlr.org/papers/volume18/16-337/16-337.pdf

    Parameters
    ----------
    collection : list of lists of persistence diagram pairs 
        list of lists (indexed by filter, image) of persistence diagram pairs (dimensions 0 and 1)
    B : integer
        The resolution parameter of vectorizing persistence diagrams. The persistence image with resolution parameter B
        is a vector of length B*B. Given two persistence diagrams, the resulting vector has length 2*B*B after concatenation.
        

    Returns
    -------
    images : a list of lists of vectors
        A list of lists (indexed by filter, image) of vectors of length 2*B*B
    """
    images = []
    persim  = gtda.diagrams.PersistenceImage(n_bins=B)
    for D in collection:
        im = persim.fit_transform(D)
        images.append(np.reshape(im,(len(im),2*B*B)))
    return images

def diagrams_to_totpers(collection):
    """Transform a list of lists of persistence diagrams (indexed by filter, image) into a list of lists
    of 2-tuples via computing total persistence

    Parameters
    ----------
    collection : list of lists of persistence diagram pairs 
        list of lists (indexed by filter, image) of persistence diagram pairs (dimensions 0 and 1)
        

    Returns
    -------
    images : a list of lists of vectors in R^2
        A list of lists (indexed by filter, image) of vectors of length 2
    """
    f = len(collection)
    s = len(collection[0])
    total_persistences = np.zeros((f,s,2))
    for i in range(f):
        for j in range(s):
            for (b,d,idx) in collection[i][j]:
                total_persistences[i,j,int(idx)]+=(d-b)**2
    return total_persistences



def concatenate_top_features(features):
    """Given a list of lists of vectors (indexed by filter, image), returns a list of vectors by concatenating along the filter dimension

    Parameters
    ----------
    features : list of lists of vectors
        Indexed by filter, image

    Returns
    -------
    a list, indexed by image, of feature vectors
    """
    return np.concatenate(features,axis = 1)

def average_top_features(features):
    """Given a list of lists of vectors (indexed by filter, image), returns a list of vectors by averaging along the filter dimension

    Parameters
    ----------
    features : list of lists of vectors
        Indexed by filter, image

    Returns
    -------
    a list, indexed by image, of feature vectors
    """
    return np.average(features,axis=0)   
