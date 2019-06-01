import os
import pandas as pd
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import roc_curve,auc,accuracy_score
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

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

def base_model(shape):
    model=Sequential()
    
    model.add(Dense(50,input_dim=shape,activation='relu'))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
    return model
              
def Classify(location_of_data,train_data_name,test_data_name,smote_or_not,purity):
    
    # smote_or_not should be either True or False to denote if Oversampling is done or not
    
    test_dataframe=pd.read_csv(location_of_data+test_data_name)
    test_data=np.array(test_dataframe)
    encoder=LabelEncoder()
    
    if smote_or_not==True:
     
        df=pd.read_csv(location_of_data+train_data_name)
    
        data=np.array(df)
        
        X=data[:,:data.shape[1]-1]
        #print(X.shape)
        Y=data[:,data.shape[1]-1]
              
        encoder.fit(Y)
        Y=encoder.transform(Y)
        Y=np_utils.to_categorical(Y)
              
        test_data_X=test_data[:,:test_data.shape[1]-1]
        test_data_Y=test_data[:,test_data.shape[1]-1]
        
        encoder.fit(test_data_Y)
        test_data_Y=encoder.transform(test_data_Y)
        test_data_Y=np_utils.to_categorical(test_data_Y)
        
        model=base_model(X.shape[1])
        model.fit(X,Y,epochs=50,batch_size=4)
        
        pred_Y=model.predict(test_data_X)
        pred_Y=(pred_Y>0.5)
              
        accuracy=accuracy_score(test_data_Y,pred_Y)
        predictions=model.predict(test_data_X)
        
        print(accuracy)
        
        return predictions
        
    
    else:
     
        df=pd.read_csv(location_of_data+train_data_name)
    
        data=np.array(df)
        
        X=data[:,:data.shape[1]-1]
        Y=data[:,data.shape[1]-1]
        
        p=0
        n=0
        
        for i in range(len(Y)):
            if Y[i]==' positive':
                p=p+1
            else:
                n=n+1
        
        pf=0#1st position of positive
        nf=0#1st position of negative
        cf=0#position to consider
        mini=''
        
        if pf<=nf:
            mini=' positive'
            cf=pf
        else:
            mini=' negative'
            cf=nf
            
        for i in range(len(Y)):
            if Y[i]==' positive':
                pf=i
                break
        
        for j in range(len(Y)):
            if Y[j]==' negative':
                nf=j
                break
        
        encoder.fit(Y)
        Y=encoder.transform(Y)
        Y=np_utils.to_categorical(Y)
        
        min_list=[]
        
        for i in range(len(Y)):
            if i==cf:
                min_list=Y[i]
                
              
              
        test_data_X=test_data[:,:test_data.shape[1]-1]
        test_data_Y=test_data[:,test_data.shape[1]-1]
        
              
        encoder.fit(test_data_Y)
        test_data_Y=encoder.transform(test_data_Y)
        test_data_Y=np_utils.to_categorical(test_data_Y)
              
        #--------------------------------------------------------------------------

        model=base_model(X.shape[1])
        model.fit(X,Y,epochs=5,batch_size=4,verbose=False)
        
        pred_Y=model.predict(test_data_X)
        pred_Y=(pred_Y>0.5)
              
        accuracy=accuracy_score(test_data_Y,pred_Y)
        predictions=model.predict(test_data_X)

        
        print("without oversampling KNN: ",accuracy)
        #--------------------------------------------------------------------------
        
        generated_samples=OverSampling.OverSampling(location_of_data,train_data_name,purity)
        
        print(generated_samples.shape)
        #print('Y shape:',Y.shape)
        
        generated_Y=generated_samples[:,generated_samples.shape[1]-1]
        '''
        encoder.fit(Y2)
        Y2=encoder.transform(Y2)
        Y2=np_utils.to_categorical(Y2)
        '''
        Y2=[]
        
        for i in range(len(generated_Y)):
            Y2.append(min_list)
        
        
        #print('y2:',Y2.shape)
        #print(Y2)
        #print('y:',Y.shape)
        #print(Y)
        balanced_X=np.concatenate((X,generated_samples[:,:generated_samples.shape[1]-1]),axis=0)
        balanced_Y=np.concatenate((Y,Y2),axis=0)
        
        '''      
        encoder.fit(balanced_Y)
        balanced_Y=encoder.transform(balanced_Y)
        balanced_Y=np_utils.to_categorical(balanced_Y)
        '''
        model=base_model(X.shape[1])
        model.fit(balanced_X,balanced_Y,epochs=50,batch_size=4,verbose=False)
        
        pred_Y=model.predict(test_data_X)
        pred_Y=(pred_Y>0.5)
              
        accuracy=accuracy_score(test_data_Y,pred_Y)
        predictions=model.predict(test_data_X)
        print("with oversampling KNN: ",accuracy)
        
        return predictions
        
        