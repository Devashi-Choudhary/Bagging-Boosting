# Bagging-Boosting

**Ensemble Learning** is a machine learning paradigm where multiple models (often called “weak learners”) are trained to solve the same problem and combined to get better results. The main hypothesis is that when weak models are correctly combined we can obtain more accurate and/or robust models.

**Bagging**, that often considers homogeneous weak learners, learns them independently from each other in parallel and combines them following some kind of deterministic averaging process.

**Boosting**, that often considers homogeneous weak learners, learns them sequentially in a very adaptative way (a base model depends on the previous ones) and combines them following a deterministic strategy.

# Dataset

The objective is to identify each of a large number of black-and-white rectangular pixel displays as one of the 26 capital letters in the English alphabet. The character images were based on 20 different fonts and each letter within these 20 fonts was randomly distorted to produce a file of 20,000 unique stimuli. Each stimulus was converted into 16 primitive numerical attributes (statistical moments and edge counts) which were then scaled to fit into a range of integer values from 0 through 15. We typically train on the first 16000 items and then use the resulting model to predict the letter category for the remaining 4000. See the [article](http://www.cs.uu.nl/docs/vakken/mpr/Frey-Slate.pdf) for more details.

**Data Set Description**

- Number of instances: 20,000
- Number of attributes: 17 (letter category and 16 numeric features)
- Attribute characteristics: Integer
- Associated Tasks: Classification
- Missing Attribute Values: None

# Dependencies

1. NumPy
2. Statistics
3. Pandas
4. Sklearn
5. Collections

# Assumptions 

1. In bagging for majority voting, the technique hard voting is used.
2. Decision Tree classifier is used with upto 2 levels of tree and 5 nodes as a weak classifier.

# How to execute the code

1. You will first have to download the repository and then extract the contents into a folder.
2. Make sure you have the correct version of Python installed on your machine. This code runs on Python 3.6 above.
3. Now, run the following command in your Terminal/Command Prompt to install the libraries required
> `pip install requirements.txt`
4. Now, you can download the dataset from [here](ftp://ftp.ics.uci.edu/pub/machine-learning-databases/letter-recognition) and put it in the current folder.
5. Open terminal. Go into the project directory folder and type the following command:
> `python Bagging.py -d <dataset path>` for Bagging Ensemble Learning.

> `python Boosting.py -d <dataset path>` for Boosting Ensemble Learning.

# Results

**1. Bagging**

![output1](https://github.com/Devashi-Choudhary/Bagging-Boosting/blob/main/Bagging.png)

In bagging we are doing random sampling with replacement then there will be chance duplicated data will be occur so there will be inconsistency among weak classifier in bagging. So  if decrease the sample size then there will be chance that accuracy will increase because probability of duplicate will be less.

**2. Boosting**

![output2](https://github.com/Devashi-Choudhary/Bagging-Boosting/blob/main/Boosting.png)

In boosting we are also performing random sampling without replacement. In every iteration the classifier is used on misclassified sample so as we increase the number of classifiers then accuracy will be increased. At some N, if we increase the classifier then accuracy will be constant this is because the sample data is fitted properly so increasing the number of classifier will not affect it.


# Acknowledgement
 This project is done as a part of college assignment.


