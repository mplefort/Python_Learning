'''
Heart Disease classification with Keras
1. Basics of classification
2. types of classification
3. Pattern recong. with Keras NN
4. Data Visualization
5. Keras Binary classifier

I. Basics
    a. Text classifier - piece of text is relevant to a topic from terms
    b. Image classifier - from a drawing, reconstruct the pattern
    c. midical classifier - from clinical data, classify a pos/neg of disease in patient or severity

II. Types of classification
    a. Supervised learning - classes/group goals known
    b. unsupervised learning - classes are unkown
    :or
    c. Parametric models - assumes class distributions follow a statistically distribution see LDA as example
    d. Nonparametric models - only takes into account distance to closest distance of obj. in space. see KNN as example.

III. Most used Algorithms for classification
    A. Naive Bayes
        - Bayes' Theorem - apriori and conditional probabilities (estimable values).
        - Text classification
    B. Gaussian Mixture models
        - Expectation maximization (EM) for training.
        - ex: train model the colors of an object and exploit this info to perform color-based tracking or segmentation
    C. Discriminant Analysis
        - describe 1-D function between two or more groups
        - supervised learning
        - ex: det. whether set of variables is effective in prediciting category membership
    D. K-nearest neighbors
        - K value > 1 parameter to look at K nearest samples and classify to group with most common nearest samples.
    E. Support Vector Machine
         -  Supervised learning, easy to analysis, good for complex data samples,
         - Ex: pattern recog, text cataloging, image classification
    F. Bayesian Decision Theory
        - P(A|B) = [ P(A) X P(B|A) ]/ P(B)


IV. Pattern Recog with Keras NN
    A.
'''

import pandas as pd

HDNames = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak',
           'slope','ca','hal','HeartDisease']

Data = pd.read_csv('cleveland_data.csv', names=HDNames)
