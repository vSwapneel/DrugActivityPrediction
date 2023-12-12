import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif,chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
import pickle


def binarize(list_val, max_val):
    binarized_list = []
    for i in range(1, max_val):
        if i in list_val :
            binarized_list.append(1)
        else :
            binarized_list.append(0)
            
    return binarized_list


# Prepare train data set
with open('Train Data.txt', 'r', encoding='utf-8') as file:
    _beta_raw_data = file.read()

raw_data = _beta_raw_data.splitlines()

#  split the array into score and data 
data_list =[]
scores_list=[]

for entry in raw_data:
    parts = entry.split('\t')
    score = int(parts[0])
    data_string = parts[1]
    data = list(map(int, data_string.split()))
    data_list.append(data)
    scores_list.append(score)

max_val =0
for element in data_list:
    max_val = max(max_val, max(element))

print(max_val)

# Binarize train data
data_list_binarized = []
for elements in data_list :
    inner_list = binarize(elements, max_val)
    data_list_binarized.append(inner_list)
    
df_data_list = pd.DataFrame(data_list_binarized)
df_scores_list = pd.DataFrame(scores_list)

with open('Train Output.txt', 'w') as output_file:
    sys.stdout = output_file
    print(scores_list)
    print(df_data_list)
    sys.stdout = sys.__stdout__

# Prepare test data
with open('Test Data.txt', 'r', encoding='utf-8') as file:
    _beta_test_data = file.read()

raw_test_data = _beta_test_data.splitlines()

test_data_list = []
for entries in raw_test_data:
    data = list(map(int, entries.split()))
    test_data_list.append(data)

# Binarize test data
test_data_binarized = []
for elements in test_data_list :
    inner_list = binarize(elements, max_val)
    test_data_binarized.append(inner_list)
    
df_test_data_list = pd.DataFrame(test_data_binarized)

# Select best features
selector = SelectKBest(chi2 , k=219)
df_data_list_selected = selector.fit_transform(df_data_list, df_scores_list)
df_test_data_list_selected = selector.transform(df_test_data_list)


def calculateLikelihood(X, y):
    alpha = 1
    num_samples, num_features = X.shape
    unique_classes = np.unique(y)
    num_classes = 2
    
    priors = np.zeros(num_classes)
    likelihoods = np.zeros((num_classes, num_features))
    
    for i, c in enumerate(unique_classes):
        priors[i] = np.mean(y == c)

        for j in range(num_features):
            count1=0
            count2=0
            for k in range(num_samples):
                if y[k]==c:
                    count2 +=1
                    if X[k,j] ==1 :
                        count1+=1
           
            likelihoods[i, j] = (count1+alpha)/(count2+alpha)
                            
    return priors, likelihoods

def predictionFunct(X, priors, likelihoods):
    num_samples, num_features = X.shape
    num_classes = len(priors)
    predictions = np.zeros(num_samples, dtype=int)
    
    for i in range(num_samples):
        posteriors = np.zeros(num_classes)
        for c in range(num_classes):
            likelihood = np.prod(likelihoods[c, :] * X[i, :] + (1 - likelihoods[c, :]) * (1 - X[i, :]))
            posteriors[c] = priors[c] * likelihood
        predictions[i] = np.argmax(posteriors)
    
    return predictions

start_time = time.time()
priors, likelihoods = calculateLikelihood(np.array(df_data_list_selected), np.array(scores_list))

Y_pred = predictionFunct(df_test_data_list_selected, priors, likelihoods)
end_time = time.time()

print("Time Ellapsed", end_time - start_time)
print(Y_pred)

with open('Pred Output.dat', 'w') as pred_output_file_path:
    sys.stdout = pred_output_file_path 
    for i, num in enumerate(Y_pred):
        if i != len(Y_pred)-1:
            pred_output_file_path.write(str(num) + "\n")
        else:
            pred_output_file_path.write(str(num))
    sys.stdout = sys.__stdout__
