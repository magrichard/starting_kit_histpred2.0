import pyreadr
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import chi2
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


## Data loading
train=pyreadr.read_r('data_train.rds') # also works for RData
test=pyreadr.read_r('data_test.rds') # also works for RData
train_df=train[None]
test_df=test[None]

## Formatting the data
#mapping={'M': 1, 'F': 0}
#train_df.replace({'sex': mapping},inplace=True)
#test_df.replace({'sex': mapping},inplace=True)

mapping2={'AD': 0, 'SC': 1}
train_df.replace({'histology': mapping2},inplace=True)

## Extract the predictor matrix X and the response vector Y
Xtrain=train_df.iloc[:,3:].copy()
Ytrain=train_df['histology'].values

Xtest=test_df.iloc[:,3:].copy()

## Try univariate logistic regression
Xvartrain=np.array(Xtrain.iloc[:,0]).reshape(-1,1)
logregmodel=LogisticRegression(solver='liblinear', random_state=0)
logregmodel.fit(Xvartrain, Ytrain)
confusion_matrix(Ytrain, logregmodel.predict(Xvartrain))

## SIS approach: multiple univariate logistic regressions
def siscreening(train_df):
    Ytrain=train_df['histology'].values
    logregmodel=LogisticRegression(solver='liblinear', random_state=0)
    pval=[]
    beta=[]
    dev=[]
    for g in range(3,train_df.shape[1]):
        Xvartrain=np.array(train_df.iloc[:,g]).reshape(-1,1)
        logregmodel.fit(Xvartrain, Ytrain)
        pval.append(1) # EDIT
        beta.append(1) # EDIT
        dev.append(1) # EDIT
    return({"pval":pval,"beta":beta,"dev":dev})
siscreening(train_df)

## Retrieve top n genes from the SIS approach
feats_n=np.argsort(siscreening(train_df)['pval'])[0:n].tolist()

## Scale the data
scaler = StandardScaler()
scaler.fit(Xtrain)
Xtrain = scaler.transform(Xtrain)
Xtest = scaler.transform(Xtest)

## Try multivariate logistic regression
feats=[0,1,2] # EDIT
Xvarstrain=Xtrain[:,feats]
logregmodel=LogisticRegression(solver='liblinear', random_state=0)
logregmodel.fit(Xvarstrain, Ytrain)
confusion_matrix(Ytrain, logregmodel.predict(Xvarstrain))

## Other classifiers
# Random Forest
from sklearn.ensemble import RandomForestRegressor
rfmodel = RandomForestRegressor(n_estimators = 1000, random_state = 0)
rfmodel.fit(Xtrain, Ytrain)

# SVM
from sklearn import svm
svmmodel = svm.SVC(kernel='linear')
svmmodel.fit(Xtrain, Ytrain)

## Cross validation
from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(Xtrain, Ytrain, test_size = 0.25, random_state = 0)
