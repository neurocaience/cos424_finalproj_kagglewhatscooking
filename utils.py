'''
This is to help automate running models
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from collections import Counter
import math
import copy as cp


def loadData():
    '''
    This function loads all the three versions of the data
    '''
    print('hi2')
    # Raw data
    data = pd.read_json('train.json')
    # TF-IDF_ingredients
    tf_ingred = pd.read_pickle('tf_idf.pkl')
    print('Train.json and tf_df.pkl loaded ...')

    # Get list of ingredients
    ingredientList = sorted(list(set([ingredient for i in range(data.shape[0]) for ingredient in data.loc[i,'ingredients']])))
    cuisineList = data['cuisine'].unique()

    print('Ingredient BOW, TF-IDF_sd...')
    # BOW_ingredients
    # the mapping between ingredient and its index
    ingredient2index = dict(zip(ingredientList, range(len(ingredientList))))
    # create a binary matrix indicating whether or not an ingredient is in a recipe
    binaryIngredientsMat = np.zeros((data.shape[0], len(ingredientList)))
    for iRecipe in range(data.shape[0]):
        binaryIngredientsMat[iRecipe, [ingredient2index[ingredient] for ingredient in data.loc[iRecipe, 'ingredients']]] = 1
        dataBinaryIngredients = pd.DataFrame(binaryIngredientsMat, columns=ingredientList)
        dataBinaryIngredients.head()
    bow_ingred = cp.deepcopy(dataBinaryIngredients.head)

    # TF-IDF_ingredients_standardized
    tf_ingred_sd = standardize(tf_ingred, ingredientList)

    print('Cuisine BOW, TF-IDF_sd...')
    # BOW cuisines
    ingredientBag = [ingredient for i in range(data.shape[0]) for ingredient in data.loc[i,'ingredients']]    
    count = []
    count.append(dict(Counter(ingredientBag)))
    for cuisine in cuisineList:
        dataCuisine = data[data['cuisine']==cuisine].copy().reset_index()
        count.append(dict(Counter([ingredient for i in range(dataCuisine.shape[0]) for ingredient in dataCuisine.loc[i,'ingredients']])))
    ingredientCount = pd.DataFrame(count).T
    ingredientCount.columns = ['total'] + [cuisine for cuisine in cuisineList]
    ingredientCount = ingredientCount.fillna(0)
    bow_cuisines = cp.deepcopy(ingredientCount)

    # TF-IDF cuisines
    # the frequency of an ingredient in each cuisine (proportion of recipes with this ingredient for a cuisine)
    ingredientFrequency = ingredientCount[[cuisine for cuisine in cuisineList]].div(data['cuisine'].value_counts()[cuisine], axis=1)
    # inverse document frequency: 1/log(proportion of cuisines that have this ingredient)
    IDF = np.log(data['cuisine'].nunique()/(ingredientCount[[cuisine for cuisine in cuisineList]]>0).sum(axis=1))
    # TF-IDF = TF*IDF
    tf_cuisines = ingredientFrequency.multiply(IDF, axis=0)

    tf_cuisines_sd = standardize(tf_cuisines, ingredientList)
    print('Done loading data')

    return data, ingredientList, cuisineList, bow_ingred, tf_ingred, tf_ingred_sd, bow_cuisines, tf_cuisines, tf_cuisines_sd

def standardize(inFile, colNames):
    # standardize a pd dataframe, input infile and column names of dataframe
    out = cp.deepcopy(inFile)
    out = StandardScaler().fit_transform(out)
    out = pd.DataFrame(out, columns=colNames)
    return out
    

def getLL(df,model):
    LL = model.score(df.values)
    LLmean = LL/df.shape[0]
    print("Log-likelihood:", LL)
    print("Average LL per data point: ", LLmean)
    return LL, LLmean

def getBIC(LL,k,n):
    '''
    Returns BIC value, give LL
    k = number of parameters
    n = number of samples
    '''
    bic = k*math.log(n) - 2*LL
    print("BIC:", bic)
    print("Average BIC per data point: ", bic/n)    
    return bic

def splitTrainingTesting(X):
    df_test = X.sample(frac=.2, random_state=3) 
    df_train = X.drop(df_test.index)
    return df_test, df_train

