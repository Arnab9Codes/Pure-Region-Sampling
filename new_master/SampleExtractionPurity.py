import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import roc_curve,auc,roc_auc_score

import matplotlib.pyplot as plt

import seaborn as sns
sns.set()

import warnings
warnings.filterwarnings('ignore')


def pure_sample_extraction(tree,X_train,Y_train,target_class,purity):
    
    leaves_no_of_samples=tree.apply(X_train)
    
    #print(leaves_no_of_samples)
    #print(X.shape)
    
    l=leaves_no_of_samples.reshape((leaves_no_of_samples.shape[0],1))
    Y=Y_train.reshape((Y_train.shape[0],1))
    
    leaves_no_samples_with_class=np.concatenate((l,Y),axis=1)
    
    d=dict()
    
    leaves=np.unique(leaves_no_of_samples)
    
    #print(leaves)
    
    leaves_copy=list(leaves)
    
    #print(leaves_copy)
    #print(len(leaves_copy))
    #print(leaves_copy[6])
    #print('-',target_class)
    
    t=[]
    t.append(target_class)
    for l in leaves:
        d[l]=[]
    
    for i in range(X_train.shape[0]):
        
        leaf=leaves_no_of_samples[i]
        
        
        
        if leaf in leaves_copy:
            #print('true')
            left=0
            right=0
            for j in range(X_train.shape[0]):
                
                if(leaf==leaves_no_of_samples[j]):
                    
                    
                    if (Y_train[j] in t):
                        right=right+1
                    else:
                        left=left+1
                     
                else:
                    
                    None
                    
            d[leaf].append(left)
            d[leaf].append(right)
            leaves_copy.remove(leaf)
        
    generate_samples=[]
    
    keys=[]
    values=[]
    
    
    #print(d.keys())

    for i in range(X_train.shape[0]):
        a=d[leaves_no_of_samples[i]][0]
        b=d[leaves_no_of_samples[i]][1]

        ratio=b/(a+b)
        
        if (ratio!=0 and  (ratio>=purity) ):
            generate_samples.append(X_train[i])
            
    generate_samples=np.array(generate_samples)
    
    return generate_samples


