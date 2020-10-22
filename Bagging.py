import math
import numpy as np
import pandas as pd
from statistics import mean
from statistics import stdev
from sklearn.tree import DecisionTreeClassifier
import collections
import operator
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--path", required = True, help = "path to input dataset")
args = vars(ap.parse_args())
Data=pd.read_csv(args['path'],header=None)

# ### Training-Testing Split

def train_test(Data):
    train_data=Data.sample(frac=0.7)
    test_data=Data.drop(train_data.index)
    return train_data,test_data
Train,Test=train_test(Data)

Train.index=range(len(Train))
Test.index=range(len(Test))

# ### Bagging Implementation

def Classifier(train,trainlabel,test):
    clf = DecisionTreeClassifier(max_depth=2,max_leaf_nodes=5, random_state=0)
    clf.fit(train,trainlabel)
    predictedtrain=clf.predict(train)
    predictedtest=clf.predict(test)
    posterior=clf.predict_proba(test)
    return predictedtrain,predictedtest,posterior


def Hard_Predicted(Label):
    predicted_label=[]
    Label=pd.DataFrame(Label)
    row,col=Label.shape
    for i in range(col):
        counter=collections.Counter(Label[i])
        sorted_counter = sorted(counter.items(), key=operator.itemgetter(0))
        max_counter=max(counter.items(), key=operator.itemgetter(1))
        y={}
        for term in counter:
            if(counter[term]>=max_counter[1]):
                y[term]=counter[term]
        sorted_counter = sorted(y.items(), key=operator.itemgetter(0))
        predicted_label.append(sorted_counter[0][0])
    return (predicted_label)


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
            
        Training_Predicted_Label,Testing_Predicted_Label,Training_Accuracy,Testing_Accuracy,posterior=Bagging(train1,actual_train1,test1,actual_test1,n)
        Accuracy.append(Testing_Accuracy)
    mean_accuracy=mean(Accuracy)
    standard_deviation=stdev(Accuracy)
    return mean_accuracy,standard_deviation

def Bagging(Train,TrainLabel,Test,TestLabel,N):
    Train=pd.DataFrame(Train)
    TrainLabel=pd.DataFrame(TrainLabel)
    Train.index=range(len(Train))
    TrainLabel.index=range(len(TrainLabel))
    TrainLabel.columns=[16]
    Train1=pd.concat([Train,TrainLabel],axis=1)
    Train1.columns=['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16']
    Predicted_Train=[]
    Predicted_Test=[]
    Predicted_Posterior=[]
    for i in range(N):
        training_subset=pd.DataFrame()
        training_subset=(Train1.sample(frac=0.7,replace=True))
        Train_Predicted_Label,Test_Predicted_Label,Posterior=Classifier(training_subset.iloc[:,0:16],training_subset['16'],Test)
        Predicted_Train.append(Train_Predicted_Label)
        Predicted_Test.append(Test_Predicted_Label)
        Predicted_Posterior.append(Posterior)
    test_hard_prediction=Hard_Predicted(Predicted_Test)
    testing_accuracy=accuracy(test_hard_prediction,TestLabel)
    return Train_Predicted_Label,test_hard_prediction,1,testing_accuracy,Predicted_Posterior



N=[3,5,10,15]
Best_Acc=[]
for i in range(len(N)):
        train=Train.drop([0],axis=1)
        train.columns=['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15']
        d=int(len(train)/5)
        train_data= list(divide_chunks((train.values.tolist()), d)) 
        training_label = list(divide_chunks(list(Train[0]), d))
        Acc,std=K_fold(train_data,training_label,Test.iloc[:,1:17],Test[0],N[i])
        print("N=",N[i])
        print("Mean Accuracy is :",Acc)
        print("Standard Deviation is :",std)
        print("Training Error is :",1-Acc)
        Best_Acc.append(Acc)
index=Best_Acc.index(max(Best_Acc))
Training_Predicted_Label,Testing_Predicted_Label,Training_Accuracy,Actual_Testing_Accuracy,Predicted_Score=Bagging(Train.iloc[:,1:17],Train[0],Test.iloc[:,1:17],Test[0],N[index])
print("Error Rate",1-Actual_Testing_Accuracy)
print("Actual Testing Accuracy",Actual_Testing_Accuracy)

