import os
import pandas as pd
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import roc_curve,auc,accuracy_score

import matplotlib.pyplot as plt

import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings('ignore')

#custom functions imported

import SampleExtractionPurity
import DecisionRules
import TreePathDictionary
import SampleGeneration
import OverSampling


def Classify(location_of_data,train_data_name,test_data_name,smote_or_not,purity):
    
    # smote_or_not should be either True or False to denote if Oversampling is done or not
    
    test_dataframe=pd.read_csv(location_of_data+test_data_name)
    test_data=np.array(test_dataframe)
    
    if smote_or_not==True:
     
        df=pd.read_csv(location_of_data+train_data_name)
    
        data=np.array(df)
        
        X=data[:,:data.shape[1]-1]
        #print(X.shape)
        Y=data[:,data.shape[1]-1]
        
        test_data_X=test_data[:,:test_data.shape[1]-1]
        test_data_Y=test_data[:,test_data.shape[1]-1]
    
        tree=DecisionTreeClassifier(random_state=9)

        tree_params={'max_features':[x for x in range(1,X.shape[1],1)],
                'max_depth':[x for x in range(1,64,1)]}

        tree_grid=GridSearchCV(tree,tree_params,scoring='accuracy',n_jobs=-1)
        tree_grid.fit(X,Y)
         
        accuracy=accuracy_score(test_data_Y,tree_grid.predict(test_data_X))
        predictions=tree_grid.predict(test_data_X)
        
        print(accuracy)
        
        return predictions
        
    
    else:
     
        df=pd.read_csv(location_of_data+train_data_name)
    
        data=np.array(df)
        
        X=data[:,:data.shape[1]-1]
        Y=data[:,data.shape[1]-1]
        
        test_data_X=test_data[:,:test_data.shape[1]-1]
        test_data_Y=test_data[:,test_data.shape[1]-1]
        
        #--------------------------------------------------------------------------
        tree_without_oversampling =DecisionTreeClassifier(random_state=9)

        tree_params={'max_features':[x for x in range(1,X.shape[1],1)],
                'max_depth':[x for x in range(1,64,1)]}

        tree_grid_without_ovrsmpl=GridSearchCV(tree_without_oversampling,tree_params,scoring='accuracy',n_jobs=-1)
        tree_grid_without_ovrsmpl.fit(X,Y)
         
        accuracy_without=accuracy_score(test_data_Y,tree_grid_without_ovrsmpl.predict(test_data_X))
        
        #y_test=tree_grid.predict(test_data)
        #predictions=tree_grid.predict(test_data_X)
        
        print("without oversampling: ",accuracy_without)
        #--------------------------------------------------------------------------
        
        generated_samples=OverSampling.OverSampling(location_of_data,train_data_name,purity)
        
        print(generated_samples.shape)
        print('Y shape:',Y.shape)
        
        
        balanced_X=np.concatenate((X,generated_samples[:,:generated_samples.shape[1]-1]),axis=0)
        balanced_Y=np.concatenate((Y,generated_samples[:,generated_samples.shape[1]-1]),axis=0)
        
        #print('balanced X:',balanced_X.shape)
        #print('balanced Y:',balanced_Y.shape)
    
        tree=DecisionTreeClassifier(random_state=9)

        tree_params={'max_features':[x for x in range(1,X.shape[1],1)],
                'max_depth':[x for x in range(1,64,1)]}

        tree_grid=GridSearchCV(tree,tree_params,scoring='accuracy',n_jobs=-1)
        tree_grid.fit(balanced_X,balanced_Y)
         
        accuracy=accuracy_score(test_data_Y,tree_grid.predict(test_data_X))
        
        #y_test=tree_grid.predict(test_data)
        predictions=tree_grid.predict(test_data_X)
        
        print("with oversampling: ",accuracy)
        
        return predictions
        
        