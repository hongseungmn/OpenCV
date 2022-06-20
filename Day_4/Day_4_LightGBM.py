from bitarray import test
from lightgbm import LGBMClassifier, train

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


dataset = load_breast_cancer()
ftr = dataset.data
target = dataset.target

X_train,X_test,y_train,y_test = train_test_split(ftr,target,test_size= 0.2,random_state=156)

lgbm_wrapper = LGBMClassifier(n_estimators=400)

evals = [(X_test,y_test)]
lgbm_wrapper.fit(X_train,y_train,early_stopping_rounds=100,eval_metric="logloss",
                 eval_set= evals,verbose=True)

preds = lgbm_wrapper.predict(X_test)
pred_proba = lgbm_wrapper.predict_proba(X_test)[:,1]


def get_clf_eval(y_test,pred):
    confusion = confusion_matrix(y_test,pred)
    accuracy = accuracy_score(y_test,pred)
    precision = precision_score(y_test,pred)
    recall = recall_score(y_test,pred)
    # F1 score 추가
    f1 = f1_score(y_test,pred)
    print("오차 행렬")
    print(confusion)
    #f1 score print 추가
    print("정확도: {0:.4f}, 정밀도: {1:.4f}, 재현률: {2:.4f}, F1: {3:.4f}".format(accuracy,precision,recall))
          
get_clf_eval(y_test,preds,pred_proba)

from lightgbm import plot_importance
import matplotlib.pyplot as plt

fig,ax = plt.subplot(figsize=(10,12))
plot_importance(lgbm_wrapper,ax=ax)