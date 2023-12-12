import sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif,chi2
from sklearn.decomposition import PCA
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


with open('Train Data.txt', 'r', encoding='utf-8') as file:
    _beta_raw_data = file.read()

raw_data = _beta_raw_data.splitlines()

data_list =[]
scores_list=[]

for entry in raw_data:
    parts = entry.split('\t')
    score = int(parts[0])
    data_string = parts[1]
    data = list(map(int, data_string.split()))
    data_list.append(data)
    scores_list.append(score)

df_scores_list = pd.DataFrame(scores_list)

with open('Data List Output.pkl', 'rb') as file:
    data_list_binarized = pickle.load(file)

df_data_list = pd.DataFrame(data_list_binarized)

def fit(X, y):
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

def predict(X, priors, likelihoods):
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


best_k = 9
max_f1_score = -1

f1_scores=[]

for k in range(9,500,7) :
    selector = SelectKBest(mutual_info_classif , k=100)
    df_data_list_selected = selector.fit_transform(df_data_list, df_scores_list)
    df_test_data_list_selected = selector.transform(df_test_data_list)

    X = df_data_list_selected
    Y = scores_list
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=42)
    
    priors_test, likelihoods_test = fit(np.array(X_train), np.array(Y_train))
    Y_pred = predict(X_test, priors_test, likelihoods_test)
    
    f1_score_var = f1_score(Y_test, Y_pred)
    print("Value of K : ", k)
    print("f1_score : ", f1_score_var)
    
    f1_scores.append(f1_score_var)
    
    max_f1_score = max(max_f1_score, f1_score_var)
    
with open('Best Features.txt', 'w') as pred_output_file_path:
    sys.stdout = pred_output_file_path 
    print(f1_scores)
    sys.stdout = sys.__stdout__
