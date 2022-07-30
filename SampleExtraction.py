
import numpy as np

from sklearn.tree import _tree
from collections import Counter

def pure_sample_extraction(tree,X_train,Y_train,target_class):
    
    leaves_no_of_samples=tree.apply(X_train)
    leaves_dictionary=Counter(leaves_no_of_samples)
    
    l=leaves_no_of_samples.reshape((leaves_no_of_samples.shape[0],1))
    Y=Y_train.reshape((Y_train.shape[0],1))
    
    leaves_no_samples_with_class=np.concatenate((l,Y),axis=1)
    
    indices_elimination=[]
    
    xtr=[]
    ytr=[]
    
    for i in range(0,len(leaves_no_samples_with_class),1):
        
        leaf_no=leaves_no_samples_with_class[i][0]
        leaf_target=leaves_no_samples_with_class[i][1]
        
        for j in range(0,len(leaves_no_samples_with_class),1):
            
            if((leaf_no==leaves_no_samples_with_class[j][0]) and (leaf_target!=leaves_no_samples_with_class[j][1])):
                
                if i not in indices_elimination:
                    
                    indices_elimination.append(i)
                    
                if j not in indices_elimination:    
                        
                    indices_elimination.append(j)
    
    leaves_samples=[]
    
    
    for i in range(0,len(leaves_no_samples_with_class),1):
        

        
        if i in indices_elimination:
            None

        else:
            xtr.append(X_train[i])
            ytr.append(Y_train[i])
            leaves_samples.append(leaves_no_samples_with_class[i])
    
    xtr=np.array(xtr)
    ytr=np.array(ytr)
    
    target_samples=[]
    target_samples_with_class=[]
    target_samples_leaves=[]
    
    leaves_samples=np.array(leaves_samples)
    
    for i in range(0,len(leaves_samples),1):
        
        leaf_no=leaves_samples[i][0]
        leaf_target=leaves_samples[i][1]
        

            
        if(leaf_target==target_class):
                
                    
            target_samples.append(xtr[i])
                    

            a=xtr[i]
            a=a.reshape((1,a.shape[0]))
            b=ytr[i]
            b=b.reshape((1,1))
                
            target_samples_with_class.append(np.concatenate((a,b),axis=1))
                    
            target_samples_leaves.append(leaves_samples[i])
    

    return target_samples,target_samples_with_class,target_samples_leaves


