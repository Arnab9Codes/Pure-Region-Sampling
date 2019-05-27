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


#custom functions imported below

import SampleExtractionPurity
import DecisionRules
import TreePathDictionary
import SampleGeneration


def OverSampling(location_of_data,train_data,purity):
    
    files=os.listdir(location_of_data)
    

    print('\nOversampling on:\n')


    #df=pd.concat(l)
    df=pd.read_csv(location_of_data+train_data)
    
    data=np.array(df)
    X=data[:,:data.shape[1]-1]
    Y=data[:,data.shape[1]-1]
    
    tree=DecisionTreeClassifier(random_state=9)

    tree_params={'max_features':[x for x in range(1,X.shape[1],1)],
            'max_depth':[x for x in range(1,64,1)]}

    tree_grid=GridSearchCV(tree,tree_params,scoring='accuracy',n_jobs=-1)
    tree_grid.fit(X,Y)
    
    Y2=list(Y)
    cls=np.unique(Y2,return_counts=True)

    minority=''

    #cls[0]=list(cls[0])
    #cls[1]=list(cls[1])
    pos=0
    neg=0
    
    for i in range(Y.shape[0]):
        if Y[i]==' positive':
            pos=pos+1
        else:
            neg=neg+1
    
    if pos<neg:
        minority=' positive'
        n=pos
        majority=' negative'
        m=neg
    else:
        minority=' negative'
        n=neg
        majority=' positive'
        m=pos
    
    print('Minority class: ', minority)
    diff=m-n
    
    best_tree=tree_grid.best_estimator_
    
    pure_samples_X_train=SampleExtractionPurity.pure_sample_extraction(best_tree,X,Y,minority,purity)
    
    p_X_train=np.array(pure_samples_X_train)
    
    #print('p_X_train: ',p_X_train.shape)
    
    dic=DecisionRules.Tree_path(best_tree,p_X_train)
    
    min_max_dic,gen_dic=TreePathDictionary.sample_generation_dictionary(dic,X)
    
    value=diff/len(gen_dic)
    samples_per_rule=round(value)
    
    generated_samples=SampleGeneration.gen_samples(gen_dic,min_max_dic,X,samples_per_rule)
    
    #print(type(generated_samples))
    
    y=[]
    for i in range(0,generated_samples.shape[0]):
        y.append(cls[0][1])
    
    y=np.array(y)
    y=y.reshape((y.shape[0],1))
    
    generated_samples=np.concatenate((generated_samples,y),axis=1)
    
    return generated_samples
