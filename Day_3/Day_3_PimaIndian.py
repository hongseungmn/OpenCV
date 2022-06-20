
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_recall_curve,roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix, precision_score,recall_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Binarizer
from sklearn.linear_model import LogisticRegression


diabetes_data = pd.read_csv("./diabetes.csv")
#print(diabetes_data['Outcome'].value_counts())
#print(diabetes_data.head(3))



#평가 지표 출력 함수
def get_clf_eval(y_test,pred=None,pred_proba=None):
    confusion = confusion_matrix(y_test,pred)
    accuracy = accuracy_score(y_test,pred)
    precision = precision_score(y_test,pred)
    recall = recall_score(y_test,pred)
    f1 = f1_score(y_test,pred)
    roc_auc = roc_auc_score(y_test,pred)
    print("오차 행렬")
    print(confusion)
    print("\n 정확도:{0:.4f}\n 정밀도:{1:.4f}\n 재현율:{2:.4f}\n F1:{3:.4f}\n AUC:{4:.4f} ".format(accuracy,precision,recall,f1,roc_auc))

#정밀도 재현율 곡선 그래프 출력 함수
def precision_recall_curve_plot(y_test,pred_proba_c1):
    precisions, recalls, thresholds = precision_recall_curve(y_test,pred_proba_c1)
    
    plt.figure(figsize=(8,6))
    threshold_boundary = thresholds.shape[0]
    plt.plot(thresholds,precisions[0:threshold_boundary],linestyle='--',label='precision')
    plt.plot(thresholds,recalls[0:threshold_boundary],label='recall')

    #threshold 값 X 축의 Scale을 0.1 단위로 변경
    start,end =plt.xlim()

    plt.xticks(np.round(np.arange(start,end,0.1),2))
   
    # x축, y축 label과 legend, 그리고 grid 설정
    plt.xlabel('Threshold value'); plt.ylabel('Precision and Recall value')
    plt.legend();plt.grid()
    plt.show()

def get_eval_by_threshold(y_test,pred_proba_c1,thresholds):
    
    for custom_threshold in thresholds:
        binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_c1)
        custom_predict = binarizer.transform(pred_proba_c1)
        print("임곗값:",custom_threshold)
        get_clf_eval(y_test,custom_predict)


#결측치 확인
#print(diabetes_data.info())

X = diabetes_data.iloc[:,:-1]
y = diabetes_data.iloc[:,-1]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.2,random_state=156,stratify=y)

#로지스틱 회귀로 학습, 예측 및 평가 수행
lr_clf = LogisticRegression()
lr_clf.fit(X_train,y_train)
pred = lr_clf.predict(X_test)
pred_proba = lr_clf.predict_proba(X_test)[:,1]

#get_clf_eval(y_test,pred,pred_proba)

pred_proba_c1 = lr_clf.predict_proba(X_test)[:,1]
#precision_recall_curve_plot(y_test,pred_proba_c1)

#print(diabetes_data.describe())

#plt.hist(diabetes_data["Glucose"],bins=10)
#plt.show()

#0값을 검사할 피처 명 리스트
zero_features = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]

#전체 데이터 건수
total_count = diabetes_data["Glucose"].count()

for feature in zero_features:
    zero_count = diabetes_data[diabetes_data[feature] == 0][feature].count()
    print("{0} 0 데이터 건수는 {1}, 퍼센트는 {2:.2f}%".format(feature,zero_count,100*zero_count/total_count))
    
#zero_features리스트 내부에 저장된 개별 피처들에 대해서 0값을 평균 값으로 대체
mean_zero_features = diabetes_data[zero_features].mean()
diabetes_data[zero_features] = diabetes_data[zero_features].replace(0,mean_zero_features)

X = diabetes_data.iloc[:,:-1]
y = diabetes_data.iloc[:,-1]

#Standardscaler 클래스를 이용해 피처 데이터 세트에 일괄적으로 스케일링 적용
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=156,stratify=y)

lr_clf = LogisticRegression()
lr_clf.fit(X_train,y_train)
pred = lr_clf.predict(X_test)
pred_proba = lr_clf.predict_proba(X_test)[:,1]

#get_clf_eval(y_test,pred,pred_proba)

thresholds = [0.3,0.33,0.36,0.39,0.42,0.45,0.48,0.50]
pred_proba = lr_clf.predict_proba(X_test)
#get_eval_by_threshold(y_test,pred_proba[:,1].reshape(-1,1),thresholds)