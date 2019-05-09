import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import math
import copy as cp
from utils import splitTrainingTesting, getLL, getBIC
from classModels import unsuperModels

'''
This script runs PCA analysis
'''

def plotPCA(principalDf, fulldf):
    finalDf = pd.concat([principalDf, fulldf['cuisine']], axis=1)
    finalDf.head()
    cuisineList = fulldf['cuisine'].unique()
    cuisineList = ['russian','british','french','italian','greek','southern_us',
              'spanish','irish',
              'moroccan','jamaican','mexican','indian','cajun_creole','brazilian',
              'chinese','japanese','korean',
              'thai','vietnamese','filipino',
              ]
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('PC 1', fontsize = 15)
    ax.set_ylabel('PC 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = cuisineList
    #colors = ['r', 'g', 'b']
    colors = []
    for i in range(len(targets)):
        colors.append([(i/len(targets), 0, .2)])
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['cuisine'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'PC 1']
               , finalDf.loc[indicesToKeep, 'PC 2']
               , c = color
               , s = 20)
    ax.legend(targets)
    ax.grid()
    plt.xlim(0, 2)    

def analyzePCA(df,fulldf,n_pcs,data_used, name):
    df_test, df_train = splitTrainingTesting(df)
    pca = PCA(n_components=n_pcs)
    principalComponents = pca.fit_transform(df_train)
    colName = []
    for i in range(principalComponents.shape[1]):
        colName.append('PC ' + str(i+1))
    principalDf = pd.DataFrame(data = principalComponents, columns = colName)
    
    # Get performance
    LL, LLmean = getLL(df_test,pca)
    getBIC(LL, n_pcs, len(df_test))
    
    # Plot PCA
    plotPCA(principalDf, fulldf)

    pca_res = unsuperModels(data_used, LL, BIC, name, n_pcs, principalDf)

    
    return pca_res

