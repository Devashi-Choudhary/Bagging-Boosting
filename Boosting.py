import collections
import math
from statistics import  mean
from statistics import stdev
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--path", required = True, help = "path to input dataset")
args = vars(ap.parse_args())
Data=pd.read_csv(args['path'],header=None)


def train_test(Data):
    train_data=Data.sample(frac=0.8)
    test_data=Data.drop(train_data.index)
    return train_data,test_data
Train,Test=train_test(Data)


Train.index=range(len(Train))
Test.index=range(len(Test))

def accuracy(predicted,actual):
    c=0
    for i in range(len(predicted)):
        if(predicted[i]==actual[i]):
            c=c+1
    s=c/len(predicted)
    return s


def divide_chunks(l, n): 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 

def K_fold(train_fold,newlabel,testing,testinglabel,n):
    Accuracy=[]
    for i in range(5):
        test1=[]
        train1=[]
        actual_test1=[]
        actual_train1=[]
        for j in range(5):
            if(i==j):
                test1=train_fold[i]
                actual_test1=newlabel[i]
            else:
                train1=train1+train_fold[j]
                actual_train1=actual_train1+newlabel[j]
            
        Training_Predicted_Label,Testing_Predicted_Label,Training_Accuracy,Testing_Accuracy,posterior=Boosting(train1,actual_train1,test1,actual_test1,n)
        Accuracy.append(Testing_Accuracy)
    mean_accuracy=mean(Accuracy)
    standard_deviation=stdev(Accuracy)
    return mean_accuracy,standard_deviation



def Classifier(train,trainlabel,test,testlabel,weight):
    clf = DecisionTreeClassifier(max_depth=2,max_leaf_nodes=5)
    clf.fit(train,trainlabel,sample_weight=weight)
    predictedtrain=clf.predict(train)
    predictedtest=clf.predict(test)
    posterior=clf.predict_proba(test)
    return predictedtrain,predictedtest,posterior

def Measure_Error(Predicted,Actual,Weight):
    error=0
    for i in range(len(Actual)):
        if(Actual[i]!=Predicted[i]):
            error+=Weight[i]
    return error


import copy
def Update_Weights(Weight,Learning_Rate,Training_Actual_Label,Training_Predicted_Label):
    Update_Weights=copy.deepcopy(Weight)
    for i in range(len(Training_Actual_Label)):
        if(Training_Actual_Label[i]==Training_Predicted_Label[i]):
            Update_Weights[i]=Weight[i]*math.exp((-1)*Learning_Rate)
        else:
            Update_Weights[i]=Weight[i]*math.exp(Learning_Rate)
    return Update_Weights       


def Hard_Predicted(Label,Learning):
    predicted_label=[]
    Label=pd.DataFrame(Label)
    row,col=Label.shape
    for i in range(col):
        label={}
        learning_label={}
        for j in range(len(Label[i])):
            if Label[i][j] in label:
                label[Label[i][j]].append(j)
            else:
                label[Label[i][j]]=[j]
        for key in label.keys():
            s=0
            for k in range(len(label[key])):
                s+=Learning[label[key][k]]
            learning_label[key]=s
        topk=(sorted(learning_label.items(), key = lambda kv:(kv[1], kv[0] ),reverse=True)) 
        predicted_label.append(topk[0][0])  
    return predicted_label


# In[12]:


import math
def Boosting(Training,Training_Actual_Label,Testing,Testing_Label,k):
    Weights=[1/(len(Training)) for i in range(len(Training))]
    Training_Predicted_Label=[]
    Testing_Predicted_Label=[]
    Predicted_Score=[]
    Learning=[]
    for i in range(k):
        Train_Predicted_Label,Test_Predicted_Label,posterior= Classifier(Training,Training_Actual_Label,Testing,Testing_Label,Weights)
        Error=Measure_Error(Train_Predicted_Label,Training_Actual_Label,Weights)
        Learning_Rate= 0.5*(math.log((1-Error)/Error))+math.log(25)
        Weights=Update_Weights(Weights,Learning_Rate,Training_Actual_Label,Train_Predicted_Label)
        s=sum(Weights)
        for i in range(len(Weights)):
            Weights[i]=Weights[i]/s
        Testing_Predicted_Label.append(Test_Predicted_Label)
        Training_Predicted_Label.append(Train_Predicted_Label)
        Learning.append(Learning_Rate)
        Predicted_Score.append(posterior)
    Training_Predicted=Hard_Predicted(Training_Predicted_Label,Learning)
    Training_Accuracy=accuracy(Training_Predicted,Training_Actual_Label)
    Testing_Predicted=Hard_Predicted(Testing_Predicted_Label,Learning)
    Testing_Accuracy=accuracy(Testing_Predicted,Testing_Label)
    return Training_Predicted_Label,Testing_Predicted_Label,Training_Accuracy,Testing_Accuracy,Predicted_Score

N=[3,10,20]
Best_Acc=[]
for i in range(len(N)):
        train=Train.drop([0],axis=1)
        train.columns=['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15']
        d=int(len(train)/5)
        train_data= list(divide_chunks((train.values.tolist()), d)) 
        training_label = list(divide_chunks(list(Train[0]), d))
        Acc,standard_deviation=K_fold(train_data,training_label,Test.iloc[:,1:17],Test[0],N[i])
        print("N =",N[i])
        print("Mean Accuracy",Acc)
        print("Training Error Rate",1-Acc)
        print("Standard Deviation",standard_deviation)
        Best_Acc.append(Acc)
        
index=Best_Acc.index(max(Best_Acc))
Training_Predicted_Label,Testing_Predicted_Label,Training_Accuracy,Actual_Testing_Accuracy,Predicted_Score=Boosting(Train.iloc[:,1:17],Train[0],Test.iloc[:,1:17],Test[0],N[index])
print("Actual Testing Accuracy",Actual_Testing_Accuracy)
print("Error Rate",1-Actual_Testing_Accuracy)



