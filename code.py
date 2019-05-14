import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import copy as cp
import pickle
import math

##### Preprocessing: five representations of data #####
data = pd.read_json('train.json')

## A: Bag of Word
# get the list of ingredients
ingredientList = sorted(list(set([ingredient for i in range(data.shape[0]) for ingredient in data.loc[i,'ingredients']])))
# the mapping between ingredient and its index
ingredient2index = dict(zip(ingredientList, range(len(ingredientList))))
# create a binary matrix indicating whether or not an ingredient is in a recipe
binaryIngredientsMat = np.zeros((data.shape[0], len(ingredientList)))
for iRecipe in range(data.shape[0]):
    binaryIngredientsMat[iRecipe, [ingredient2index[ingredient] for ingredient in data.loc[iRecipe, 'ingredients']]] = 1
dataBinaryIngredients = pd.DataFrame(binaryIngredientsMat, columns=ingredientList)
dataBinaryIngredients.head()

## B: TF-IDF on BOW
# extract the value of the matrix to a numpy array - to speed up calculation
X = dataBinaryIngredients.values
# term frequency
TF = X/np.sum(X,axis=1)[:,np.newaxis]
# inverse document frequency: 1/log(proportion of recipes that have this ingredient)
IDF = np.log(dataBinaryIngredients.shape[0]/X.sum(axis=0)[np.newaxis,:])
# TF-IDF = TF*IDF, and create dataframe
TFIDF = pd.DataFrame(TF*IDF, index=dataBinaryIngredients.index, columns=dataBinaryIngredients.columns)

## C: Ingredient Frequency
count = []
count.append(dict(Counter(ingredientBag)))
for cuisine in cuisineList:
    dataCuisine = data[data['cuisine']==cuisine].copy().reset_index()
    count.append(dict(Counter([ingredient for i in range(dataCuisine.shape[0]) for ingredient in dataCuisine.loc[i,'ingredients']])))
ingredientCount = pd.DataFrame(count).T
ingredientCount.columns = ['total'] + [cuisine for cuisine in cuisineList]
ingredientCount = ingredientCount.fillna(0)
ingredientFrequency = ingredientCount[[cuisine for cuisine in cuisineList]].div(data['cuisine'].value_counts()[cuisine], axis=1)

## D: TF-IDF on cuisines
# the frequency of an ingredient in each cuisine (proportion of recipes with this ingredient for a cuisine)
ingredientFrequency = ingredientCount[[cuisine for cuisine in cuisineList]].div(data['cuisine'].value_counts()[cuisine], axis=1)
# inverse document frequency: 1/log(proportion of cuisines that have this ingredient)
IDF = np.log(data['cuisine'].nunique()/(ingredientCount[[cuisine for cuisine in cuisineList]]>0).sum(axis=1))
# TF-IDF = TF*IDF
ingredientTFIDF = ingredientFrequency.multiply(IDF, axis=0)

## E: BOW + one-hot encoding of cuisines
data2 = data.drop(['id','ingredients'], axis=1)
data3 = data2.join(dataBinaryIngredients)
categorical_columns = ['cuisine']
BOWwCuisine = pd.get_dummies(data3, columns = categorical_columns)


##### Data Exploration #####

## Basic information of cuisines and ingredients
# the list of cuisines
NCuisine = data['cuisine'].nunique()
cuisineList = data['cuisine'].unique()
print('All cuisines:',[cuisine for cuisine in cuisineList])

# the number of recipes in each cuisine
data['cuisine'].value_counts()

# the list of ingredients
ingredientBag = [ingredient for i in range(data.shape[0]) for ingredient in data.loc[i,'ingredients']]
ingredientList = sorted(list(set(ingredientBag)))
NIngredient = len(ingredientList)
print('In total:', str(NIngredient), 'ingredients')


## Characterizing the cuisines (common and signature ingredients)
# The most common ingredients in each cuisine
print('Top 10 common ingredients in each cuisine:')
Ntop = 10
for cuisine in cuisineList:
    print(cuisine,': ', ingredientFrequency.nlargest(Ntop, cuisine).index.values)

# The most "signature" ingredients in each cuisine, i.e. ingredients with top TF-IDF in each cuisine
print('Top 10 *signature* ingredients in each cuisine (according to TF-IDF):')
Ntop = 10
for cuisine in cuisineList:
    print(cuisine,': ', ingredientTFIDF.nlargest(Ntop, cuisine).index.values)


## Relationship between cuisines: Cuisine similarity, based on ingredient frequency (correlation between ingredient frequency vectors)
import scipy.cluster.hierarchy as sch
from mpl_toolkits.axes_grid1 import make_axes_locatable

corrmat = ingredientFrequency.corr().values
d = sch.distance.pdist(corrmat)
L = sch.linkage(d, method='complete')
ind = sch.fcluster(L, 0.5*d.max(), 'distance')
columns = [cuisineList[i] for i in list(np.argsort(ind))]

fig, ax = plt.subplots(figsize=(8,8))
ax.set_title('Cuisine similarity', fontsize=15)
p = ax.matshow(ingredientFrequency[columns].corr(), vmin=0, vmax=1)
ax.set_xticks(range(NCuisine))
ax.set_xticklabels(labels=columns, rotation=90, fontsize=14)
ax.xaxis.tick_bottom()
ax.set_yticks(range(NCuisine))
ax.set_yticklabels(labels=columns, fontsize=14)
fig.colorbar(p)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.25)
cbar = plt.colorbar(p, cax=cax)
cbar.ax.tick_params(labelsize=14)
plt.show()


##### Unsupervised Learning #####

## Truncated SVD
def splitTrainingTesting(X):
    df_test = X.sample(frac=.2, random_state=3)
    df_train = X.drop(df_test.index)
    return df_test, df_train
df_test, df_train = splitTrainingTesting(dataBinaryIngredients)
Xtrain = df_train.values
Xtest = df_test.values
# run svd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=2)
svdComps = svd.fit_transform(Xtrain)
svdCompsTest = svd.transform(Xtest)
svdRescaled = TruncatedSVD(n_components=2)
svdCompsRescaled = svdescaled.fit_transform(XtrainRescaled)
svdCompsRescaledTest = svdRescaled.transform(XtestRescaled)
# 2 component SVD, train, not rescaled
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlim(xmin=-1, xmax=4)
ax.set_ylim(ymin=-2, ymax=3)
ax.set_xlabel('SVD Component 1', fontsize = 15)
ax.set_ylabel('SVD Component 2', fontsize = 15)
ax.set_title('2 component SVD, train, not rescaled', fontsize = 20)
targets = ['salt', 'no salt']
for i in range(len(targets)):
    ax.scatter(svdComps[labelsTrain[i],0], svdComps[labelsTrain[i],1], s = 50)
ax.legend(targets)
ax.grid()
# 2 component SVD, train, not scaled
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlim(xmin=-15, xmax=20)
ax.set_ylim(ymin=-10, ymax=25)
ax.set_xlabel('SVD Component 1', fontsize = 15)
ax.set_ylabel('SVD Component 2', fontsize = 15)
ax.set_title('2 component SVD, train, rescaled', fontsize = 20)
targets = ['salt', 'no salt']
for i in range(len(targets)):
    ax.scatter(svdCompsRescaled[labelsTrain[i],0], svdCompsRescaled[labelsTrain[i],1], s = 50)
ax.legend(targets)
ax.grid()


## k-means
def splitTrainingTesting(X):
    df_test = X.sample(frac=.2, random_state=3)
    df_train = X.drop(df_test.index)
    return df_test, df_train
df_test, df_train = splitTrainingTesting(dataBinaryIngredients)
Xtrain = df_train.values
Xtest = df_test.values
#k-means with euclidean distance
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
length = 30
arr = np.zeros(length-1)
arr1 = np.zeros(length-1)
arr2 = np.zeros(length-1)
xxx = np.array(range(1, length))
for n_clusters in range(1, length):
    clusterer = KMeans(n_clusters=n_clusters)
    scores = cross_val_score(clusterer, Xtrain, cv=5)
    arr[n_clusters-1] = scores.mean()
    arr1[n_clusters-1] = scores.mean() - scores.std() * 2
    arr2[n_clusters-1] = scores.mean() + scores.std() * 2
plt.plot(xxx, arr, xxx, arr1, xxx, arr2)
fig.suptitle('k means', fontsize=16)
plt.xlabel('number of clusters k', fontsize=12)
plt.ylabel('score from 5-fold CV', fontsize=12)
# kmeans with k=2 has separated the group into salt and no-salt; can change k to larger numbers
clusterer = KMeans(n_clusters=2)
cluster_labels = clusterer.fit_predict(X)
print(cluster_labels.shape)
numInCluster1 = np.sum(cluster_labels)
numInCluster0 = cluster_labels.shape[0] - numInCluster1
numNoSaltInCluster1 = 0
numSaltInCluster0 = 0
count=0
for i in train_ids:
    if cluster_labels[count] == 1 and 0 == df_train.loc[i,'salt']:
        numNoSaltInCluster1 += 1
    if cluster_labels[count] == 0 and 1 == df_train.loc[i,'salt']:
        numSaltInCluster0 += 1
    count += 1
print(numNoSaltInCluster1/numInCluster1*100, numSaltInCluster0/numInCluster0*100)
#interpret the results of kmeans: look at users closest to each center
centers = clusterer.cluster_centers_
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=0.1)
new_centers = sel.fit_transform(centers)
print(new_centers.shape)
for i in range(df_train.shape[1]-1):
    if sel.get_support()[i]:
        print(df_train.iloc[:,i].name)

## Market Basket Analysis: https://pbpython.com/market-basket-analysis.html
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
frequent_itemsets = apriori(dataBinaryIngredients, min_support = 0.005, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
print("shape of rules from training", rules.shape)
# rules sorted by "lift"
print("rules from training based on lift")
rules.sort_values('lift').iloc[::-1, :]
# rules sorted by "confidence"
print("rules from training based on confidence")
rules.sort_values('confidence').iloc[::-1, :]
# association rules between ingredients AND also between ingredients and type of cuisine
frequent_itemsets = apriori(BOWwCuisine, min_support = 0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
print("shape of rules from training", rules.shape)
#display rules
print("rules from training based on confidence, allcuisines")
rules.sort_values('confidence').iloc[::-1, :]

## LDA
# six latent components (latent recipes); change NComponents to get results for other number of components
NComponents = 6
lda = LatentDirichletAllocation(n_components = NComponents)
lda.fit(TFIDF.values)
recipeProportionsfromLatentRecipes = lda.transform(dataBinaryIngredients.values)
latentRecipes = lda.components_
# top (0.1%) ingredients for each "latent recipe"
print('Top ingredients for each "latent recipe":')
for i in range(NComponents):
    topIndices = np.where(latentRecipes[i,:] > np.percentile(latentRecipes[i,:], 99.9))[0]
    topIndices = topIndices[np.argsort(-latentRecipes[i, topIndices])]
    print('Latent recipe', i+1, ':', TFIDF.columns.values[topIndices], latentRecipes[i, topIndices])


## BMF
# six latent components (latent recipes); change NComponents to get results for other number of components
NComponents = 6
X = dataBinaryIngredients.values
# use default values for lambda_w and lambda_h, increase max_iter to 100 to make sure fitting converges
bmf = nimfa.Bmf(X, seed="nndsvd", rank=NComponents, max_iter=100, lambda_w=1.1, lambda_h=1.1)
bmf_fit = bmf()
# get the two matrices
W = np.array(bmf_fit.coef())
H = np.array(bmf_fit.basis())
# top (0.1%) ingredients for each "latent recipe"
print('Top ingredients for each "latent recipe":')
Ntop = 10
for i in range(NComponents):
    topIndices = np.where(W[i,:] > np.percentile(W[i,:], 99.9))[0]
    topIndices = topIndices[np.argsort(-W[i, topIndices])]
    # print the ingredients and their weights
    print('Latent recipe', i+1, ':', dataBinaryIngredients.columns.values[topIndices], W[i, topIndices])


## PCA
X = TFIDF
pca_TFIDF = PCA(n_components=NComponents)
X_r_TFIDF = pca_TFIDF.fit(X).transform(X)
# Percentage of variance explained for each components
print('explained variance ratio: %s' % str(pca_TFIDF.explained_variance_ratio_))
print('explained variance ratio (cumulative): %s' % str(np.cumsum(pca_TFIDF.explained_variance_ratio_)))
# plot the center (mean) for each cuisine (width/height of the ellipses are 0.1 standard deviation of all recipes in this cuisine)
from matplotlib.patches import Ellipse
datawPCA = data.copy()
datawPCA['PC1proj'] = X_r_TFIDF[:,0]
datawPCA['PC2proj'] = X_r_TFIDF[:,1]
cuisineCenter = datawPCA.groupby('cuisine').mean()[['PC1proj','PC2proj']]
cuisineSD = datawPCA.groupby('cuisine').std()[['PC1proj','PC2proj']]
fig, ax = plt.subplots(figsize=(6,6))
for cuisine in cuisineList:
    ellipse = Ellipse(xy=(cuisineCenter.loc[cuisine,'PC1proj'], cuisineCenter.loc[cuisine,'PC2proj']), width=cuisineSD.loc[cuisine,'PC1proj']/5, height=cuisineSD.loc[cuisine,'PC2proj']/5)
    ax.add_artist(ellipse)
    plt.annotate(cuisine, (cuisineCenter.loc[cuisine,'PC1proj'], cuisineCenter.loc[cuisine,'PC2proj']+0.005), fontsize=14)
    ax.set(xlim=(-0.09,0.18), ylim=(-0.14,0.23))
    ax.set_xlabel('PC1', fontsize=14, labelpad=10)
    ax.set_ylabel('PC2', fontsize=14, labelpad=5)
    ax.tick_params(labelsize=14)
# Distance on the PC1-PC2 Plane ~ Pearson Correlation Coefficient
fig, ax = plt.subplots(figsize=(6,6))
for cuisine1 in cuisineList:
    for cuisine2 in cuisineList:
        ax.scatter(corrmat.loc[cuisine1,cuisine2], distance.loc[cuisine1,cuisine2], color='C0')
ax.set_xlabel('Pearson Correlation Coefficient', fontsize=14, labelpad=10)
ax.set_ylabel('Distance on the PC1-PC2 Plane', fontsize=14, labelpad=5)
ax.tick_params(labelsize=14)


##### Supervised Learning #####

from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from itertools import count
import seaborn as sn
import pandas as pd
from matplotlib.pyplot import figure
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def splitTrainingTesting(X):
    """Split into trained and test sets"""
    df_test = X.sample(frac=.2, random_state=3)
    df_train = X.drop(df_test.index)
    return df_test, df_train

## get training and testing data
test, train = splitTrainingTesting(dataBinaryIngredients)
train_ids = train.index.values
test_ids = test.index.values
trainLabels = data.loc[train_ids, 'cuisine'].values
testLabels = data.loc[test_ids, 'cuisine'].values

# Initial values
X_train = train
y_train =trainLabels
X_test = test
y_test = testLabels
target_cuisines = ['greek', 'southern_us', 'filipino', 'indian', 'jamaican',
                   'spanish', 'italian', 'mexican', 'chinese', 'british', 'thai',
                   'vietnamese', 'cajun_creole', 'brazilian', 'french', 'japanese',
                   'irish', 'korean', 'moroccan', 'russian']

def fct(s):
    return target_cuisines.index(s)
fct = np.vectorize(fct)

## Create functions to run the classifiers and store results in 1 variable

# Helpfer function for cross validation
def calcCrossval(model):
    scores = cross_val_score(model, X_train, y_train, cv=3)
    print("Cross Val Training Accuracy: " + str(round(scores.mean(),3)) + ", STD: " + str(round(scores.std()*2,3)))
    return scores

# Create a class that contains a model and all its variables
# In this case, the 'class' is just a variable of a model
# that contains the model's relevant variables such as fpr, tpr, etc.

class modDetails:
    # This creates an umbrella variable that stores evaluation metrics, and relevant data for each
    # classifier:
    def __init__(self, predictionsTrain, predictionsTest,
                 classifier, train_cvscores, training_accuracy, test_accuracy, clf_report,
                 cm, name):
        self.predictionsTrain = predictionsTrain
        self.predictionsTest = predictionsTest
        self.training_accuracy = training_accuracy
        self.test_accuracy = test_accuracy
        self.classifier = classifier
        self.train_cvscores = train_cvscores
        self.clf_report =  clf_report
        self.cm = cm
        self.name = name

def runModel2(model,name):
    print(name + '-'*50 + ' \n')
    
    # Train the Classifier
    # And obtain 10-fold cross validation results of the training
    
    # GMM not used in this implementation.
    if name.find('GMM') >= 0: # if this is GMM, only fit with X_train
        tt = model.fit(X_train)
    else:
        tt = model.fit(X_train,y_train)

    train_cvscores = calcCrossval(model)

    print(str(type(tt)))


    # Calculate classifier accuracy on trained data and test data
    predictionsTrain = tt.predict(X_train)
    predictionsTest = tt.predict(X_test)


training_accuracy = accuracy_score(y_train, predictionsTrain)
test_accuracy = accuracy_score(y_test, predictionsTest)

    cm = confusion_matrix(y_test, predictionsTest)
    clf_report = classification_report(y_test, predictionsTest, target_names=target_cuisines)
    
    print('Training accuracy: ', training_accuracy)
    print('Test accuracy: ', test_accuracy)
    
    
    
    # Save results in one umbrella variable of the class modDetails
    model_info = modDetails(predictionsTrain, predictionsTest,
                            tt, train_cvscores, training_accuracy, test_accuracy, clf_report, cm,
                            name)
    return model_info

## Feature Selection

def get_feature_names(selector, dataBinaryIngredients):
    """
        Returns feature names from array of indices
        selector: selectKBest
        dataBinaryIngredients: dataframe
        """
    mask = selector.get_support(indices=True) #list of booleans
    column_names = dataBinaryIngredients.columns
    feature_names = column_names[mask].values
    return feature_names

from sklearn.feature_selection import RFE
# recursive feature elimination (RFE) ### Takes a lot of time
def fs(model):
    selector = RFE(model, 10, step=1)
    selector.fit(X_train, y_train)
    print('10 most significant variables, with corresponding coefficients')
    index = 0
    for i in range(len(selector.support_)):
        if selector.support_[i]:
            print(i, selector.estimator_.coef_[0][index])
            index = index + 1

from sklearn.feature_selection import SelectKBest, chi2
selector = SelectKBest(chi2, k=20)
selector.fit_transform(X_train, y_train)
list(get_feature_names(selector, dataBinaryIngredients))

## Dimensionality Reduction
# Reset to initial values.
X_train = train
y_train =trainLabels
X_test = test
y_test = testLabels

# Truncated data

#pca, source: https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
#https://support.minitab.com/en-us/minitab/18/help-and-how-to/modeling-statistics/multivariate/how-to/principal-components/interpret-the-results/key-results/
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD#actually using truncated svd instead of pca -- X is too large and sparse for PCA

# number of components
n_components = 1000
#rescale???
scaler = StandardScaler()
XtrainRescaled = scaler.fit_transform(X_train)
XtestRescaled = scaler.transform(X_test)
print('done rescaling')

pca = TruncatedSVD(n_components= n_components)
principalComps = pca.fit_transform(X_train) # new X_train
principalCompsTest = pca.transform(X_test) # new X_test
print('done principal components')

pcaRescaled = TruncatedSVD(n_components=n_components)
principalCompsRescaled = pcaRescaled.fit_transform(XtrainRescaled)
principalCompsRescaledTest = pcaRescaled.transform(XtestRescaled)
print('done principal components rescaled')

## Run Models
# Set data to reduced dims, otherwise takes too long
print("Initial X_train.shape ", X_train.shape)
print("Initial y_train.shape ", y_train.shape)
print("Initial X_test.shape ", X_test.shape)
print("Inital y_test.shape ", y_test.shape)

X_train = principalComps
X_test = principalCompsTest
print("New X_train.shape ", X_train.shape)
print("New X_test.shape ", X_test.shape)

# 1. Logistic Regression
logregL1 = LogisticRegression(C = 1, penalty = 'l2', multi_class = 'multinomial', solver = 'lbfgs')
logregL1_info = runModel2(logregL1, 'LogRegL1')
# Visualise
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
# Plot confusion matrix: https://flothesof.github.io/kaggle-whats-cooking-machine-learning.html?fbclid=IwAR2OXhDXtZ4LRf_K7TO6x2L2d6T3O6-XNtQ1Y4mqTUB1BH7yqGCkaVWiWjM
plt.figure(figsize=(10, 10))

cm = logregL1_info.cm # confusion_matrix(y_test, predictionsTest)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.imshow(cm_normalized, interpolation='nearest')
plt.title("confusion matrix")
plt.colorbar(shrink=0.3)
cuisines = target_cuisines
tick_marks = np.arange(len(cuisines))
plt.xticks(tick_marks, cuisines, rotation=90)
plt.yticks(tick_marks, cuisines)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')

# Classification report: Adapted from https://flothesof.github.io/kaggle-whats-cooking-machine-learning.html?fbclid=IwAR2OXhDXtZ4LRf_K7TO6x2L2d6T3O6-XNtQ1Y4mqTUB1BH7yqGCkaVWiWjM
print(logregL1_info.clf_report)
# chart: accuracy, f1 score

# 2. XGBoost
from xgboost import XGBClassifier
xgb = XGBClassifier(objective="multi:softprob", random_state=0)
xgb_info = runModel2(xgb, "XG Boost")

# 3. RandomForest
rf = RandomForestClassifier(n_estimators=1000, max_depth=2, random_state=0)
rf_info = runModel2(rf, 'Random Forest')

# 4. Gradient Boosting
gbc = GradientBoostingClassifier(n_estimators=1000, max_depth=2, random_state=0)
gbc_info = runModel2(gbc, 'Gradient Boost')
