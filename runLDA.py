import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
import math


'''
Create functions for LDA analysis

Input 
- pd dataframe
- number of latent components


Output:
- Plot of Proportion of Latent User i
- Main features of Latent User i
- LL

'''

def plotUserProportions(user_proportions_from_latent_users):
    plt.figure()
    for i in range(user_proportions_from_latent_users.shape[1]):  
        plt.hist(user_proportions_from_latent_users[:, i], alpha=0.3,
            label="Latent User " + str(i+1),
            range=(0,1), bins=20)
    plt.xlabel("User Proportion from Latent User i", fontsize=20)
    plt.ylabel("Count", fontsize=20)
    plt.tick_params(labelsize=15)
    plt.legend()
    plt.show()

def plotLUPseudocounts(lu,i):
    print("Percentiles of feature pseudocounts for latent user :" + str(i))
    #print(np.percentile(lu,np.arange(0,100,1)))
    plt.figure()
    plt.hist(lu)
    plt.xlabel("Feature Pseudocount for latent user " + str(i), fontsize=20)
    plt.ylabel("Freuqency", fontsize=20)
    plt.tick_params(labelsize=15)
    plt.show()
    
def getLUFeatures(df, lu, i):
    lu_indices = np.where(lu > np.percentile(lu,99))[0]
    label = 'LU' + str(i) + 'Pseudocount'  # column name
    top_lu_df = pd.DataFrame({'Feature': 
                     df.columns.values[lu_indices],
                    label: lu[lu_indices]})
    top_lu_df.sort_values(label,inplace=True, ascending=False)
    print('Latent User', str(i), 'Top Features:')
    print(top_lu_df.head(n=10))
    return top_lu_df

def getLL(df,model):
    LL = model.score(df.values)
    LLmean = LL/df.shape[0]
    print("Log-likelihood:", LL)
    print("Average LL per data point: ", LLmean)
    return LL, LLmean

def getBIC(LL,k=5,n=11989):
    '''
    Returns BIC value, give LL
    k = number of parameters
    n = number of samples
    '''
    bic = k*math.log(n) - 2*LL
    print("BIC:", bic)
    print("Average BIC per data point: ", bic/n)    
    return bic

def getPerp(df, model):
    perp = model.perplexity(df)
    print('Perplexity', perp)
    return perp

def splitTrainingTesting(X):
    df_test = X.sample(frac=.2, random_state=3) 
    df_train = X.drop(df_test.index)
    return df_test, df_train

def analyzeLDA(df, num_components, analyze_latent_user=0):
    df_test, df_train = splitTrainingTesting(df)
    lda = LatentDirichletAllocation(n_components = num_components)
    lda.fit(df_train.values)
    # print(lda)
    LL, LLmean = getLL(df_test,lda)
    getBIC(LL, num_components, len(df_test))
    getPerp(df_test, lda)
    
    latent_users = lda.components_
    user_proportions_from_latent_users = lda.transform(df_test.values)
    print("Latent Users Shape: ", latent_users.shape)
    print("User Proportion from Latent Users Shape:", user_proportions_from_latent_users.shape)
    
    plotUserProportions(user_proportions_from_latent_users)
    
    if analyze_latent_user == 1:
    # Analyze each latent user: 
        for i in range(user_proportions_from_latent_users.shape[1]):
            # Lu is the vector of "pseudocounts": number of times a feature 
            # would be '1' for latent User 1 (lu)

            lu = latent_users[i,:]
            print('Latent User', str(i), 'Shape:', str(lu.shape))
            #plotLUPseudocounts(lu, i)
            getLUFeatures(df_test,lu,i)

def analyzeLDA_wTWP(df, num_components, analyze_latent_user=0, twp=.2):
    # twp = topic word prior 
    df_test, df_train = splitTrainingTesting(df)
    lda = LatentDirichletAllocation(n_components = num_components, topic_word_prior=twp)
    lda.fit(df_train.values)
    # print(lda)
    LL, LLmean = getLL(df_test,lda)
    getBIC(LL, num_components, len(df_test))
    getPerp(df_test, lda)
    
    latent_users = lda.components_
    user_proportions_from_latent_users = lda.transform(df_test.values)
    print("Latent Users Shape: ", latent_users.shape)
    print("User Proportion from Latent Users Shape:", user_proportions_from_latent_users.shape)
    
    plotUserProportions(user_proportions_from_latent_users)
    
    if analyze_latent_user == 1:
    # Analyze each latent user: 
        for i in range(user_proportions_from_latent_users.shape[1]):
            # Lu is the vector of "pseudocounts": number of times a feature 
            # would be '1' for latent User 1 (lu)

            lu = latent_users[i,:]
            print('Latent User', str(i), 'Shape:', str(lu.shape))
            #plotLUPseudocounts(lu, i)
            getLUFeatures(df_test,lu,i)
