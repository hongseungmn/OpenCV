from lightgbm import train
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

cust_df = pd.read_csv("/Users/hongseongmin/머신러닝/Day_4/Santander_customer_satisfaction/train2.csv",encoding='latin-1')
# print('datashape:',cust_df.shape)
# print(cust_df.head(3))
# print(cust_df.info())

# print(cust_df['TARGET'].value_counts())
# unsatisfied_cnt = cust_df[cust_df['TARGET'] == 1].TARGET.count()
# total_cnt = cust_df.TARGET.count()
# print('unsatisfied 비율은 {0:.2f}'.format((unsatisfied_cnt/total_cnt)))

#print(cust_df.describe())

cust_df['var3'].replace(-999999,2,inplace=True)
cust_df.drop('ID',axis=1,inplace=True)

X_features = cust_df.iloc[:,:-1]
y_labels = cust_df.iloc[:,-1]
# print('피처 데이터 shape:{0}'.format(X_features.shape))

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X_features,y_labels,test_size=0.2,random_state=0)

train_cnt = y_train.count()
test_cnt = y_test.count()

# print('학습세트 shape:{0}, 테스트 shape:{1}'.format(X_train.shape,X_test.shape))
# print('학습 세트 레이블 값 분포 비율')
# print(y_train.value_counts()/train_cnt)
# print('\n  테스트 세트 레이블 값 분포 비율')
# print(y_test.value_counts()/test_cnt)

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

xgb_clf = XGBClassifier(n_estimators=500,random_state=156)

xgb_clf.fit(X_train,y_train,early_stopping_rounds=100,eval_metric='auc',eval_set=[(X_train,y_train),(X_test,y_test)])

xgb_roc_score = roc_auc_score(y_test,xgb_clf.predict_proba[:,1],average='macro')
print('ROC_AUC:{0:.4f}'.format(xgb_roc_score))