import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import math
import copy as cp

'''
This plot creates a class for storing unsupervised learning model results
'''

class unsuperModels:
    '''
    This creates an umbrella variable that stores:
    data_used, LL, BIC, model_name, num_clusters, param
    '''

    def __init__(self, data_used, LL, BIC,
                 name, num_clusters, params):
        self.data_used = data_used
        self.LL = LL
        self.BIC = BIC
        self.name = name
        self.num_clusters = num_clusters
        self.params = params   # Where params = other params relevant to model
        

class superModels:
    '''
    This creates an umbrella variable that stores:
    data_used, LL, BIC, model_name, num_clusters, param
    '''

    def __init__(self, data_used, cfm_train, cfm_test,
                 accTrain, accTest, precision, recall, f1,
                 name, params):
        self.data_used = data_used
        self.cfm_train = cfm_train
        self.cfm_test = cfm_test
        self.accTrain = accTrain
        self.accTest = accTest
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.name = name
        self.params = params   # Where params = other params relevant to model
        
