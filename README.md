# convpers

This repo contains code for computing the Convolutional Persistence Tranform, vectorizing it, and running predictive models with the resulting features. The experiments/ folder contains ipynb notebooks for five experiments:
- Digits Dataset (sklearn)
- MNIST Dataset (sklearn)
- Devanagari Character Set (https://www.kaggle.com/datasets/rishianand/devanagari-character-set?resource=download)
- Chinese Digits Dataset (https://www.kaggle.com/code/rkuo2000/chinese-mnist)
- Solutions to Kuramoto-Sivashinsky PDEs

Within experiments/ is a sub-folder, convpers/, which acts as a package for computing and working with convolutional persistence. It contains the following modules:
- CPT : For computing the convolutional persistence transform of a set of images given a set of filters
- Filters: For generating standard filters, random filters, and Eigenfilters
- Data: For generating solutions to Kuramoto-Sivashinsky PDEs and preparing the chinse digits dataset
- Vectorize: For vectorizing the output of the CPT
- Testing: For running the simple classification models considered in the paper, nearest-neighbors, XGBoost, and neural networks.

This repo accompanies the paper (cite paper here). Working through one of the ipynb notebooks should provide a full explanation of how to use the convpers/ package.

Written by: Elchanan Solomon
