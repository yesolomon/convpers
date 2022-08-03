import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.decomposition import PCA 

def gaussian_kernel(l=5, sig=1.):
    """Creates a Gaussian lxl kernel with sigma=1
    Code taken from: https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy

    Parameters
    ----------
    l : int
        side length of kernel
    sig : positive real number
        variance of Gaussian

    Returns
    -------
    kernel : lxl numpy array

    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def sharpening_kernel(l):
    """Creates a lxl sharpening kernel, with all entries -1 except for those in the middle, which are positive, and whose
    value is set to make the sum of kernel entries be 0. When l is odd, the middle is a single entry, otherwise for l even we take the middle
    to be a 2x2 block

    Parameters
    ----------
    l : int
        side length of kernel

    Returns
    -------
    A : lxl numpy array

    """
    if l%2 == 1:
        A = np.ones((l,l))*-1
        i = int((l-1)/2)
        A[i,i]+=l*l
    else:
        A = np.ones((l,l))*-1
        i = int(l/2)
        A[i,i]+=(l*l)/4
        A[i,i-1]+=(l*l)/4
        A[i-1,i]+=(l*l)/4
        A[i-1,i-1]+=(l*l)/4
    return A

def blur_kernel(l):
    """Creates a lxl blurring kernel, with all 1/(l^2).

    Parameters
    ----------
    l : int
        side length of kernel

    Returns
    -------
    lxl numpy array

    """
    return np.ones((l,l))/(l*l)


def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

def random_filters(l=3,n=1):
    """Creates a list of random filters

    Parameters
    ----------
    l : int
        side length of filters
    n: int
        number of random filters

    Returns
    -------
    list of n filters of shape (l,l)

    """
    filters = []
    for i in range(n):
        filters.append(np.reshape(sample_spherical(1,l*l),(l,l)))
    return filters


def PCA_eigenfilters(X,l=3,n=1):
    """Returns principal components of set of local patches in a data set

    Parameters
    ----------
    X: list of numpy arrays
        list of images to sample patches from
    l : int
        side length of patches/filters
    n: int
        number of principal components to return

    Returns
    -------
    list of n filters of shape (l,l)

    """
    patches = []
    for im in X:
        patches.extend(extract_patches_2d(im,(l,l)))

    patches = np.reshape(patches,(len(patches),l*l))
    pca = PCA(n_components=n)
    pca.fit(patches)
    filters = []
    for i in range(n):
        filters.append(np.reshape(pca.components_[i],(l,l)))
    return filters

def random_linear_comb(filters,n=3):
    """Given a list of filters, generate n random normalized linear combinations
    normalized here means that weight vectors live on the sphere.
    ASSUMPTION: all filters are the same shape.

    Parameters
    ----------
    filters : list of numpy array
        list of filters, assumed to all be of the same shape
    n: int
        number of linear combination filters to generate

    Returns
    -------
    list of n filters 

    """
    new_filters = []
    for i in range(n):
        F = np.zeros(filters[0].shape)
        w = sample_spherical(1,len(filters))
        for j in range(len(filters)):
            F+=w[j]*filters[j]
        new_filters.append(F)
    return new_filters
            

