import matplotlib
import numpy as np
from sklearn.base import BaseEstimator # 사용자만의 변환기를 만들어 변환 파이프라인 생성
                                       # 수동으로 정제해온 작업(누락된 데이터 정제, 새로운 특성 추가, 텍스트와 범주형 특성 다루기)
class MyDummyClassifier(BaseEstimator): 
    
    def fit(self, X , y=None):
        pass
    
    def predict(self,X):
        pred = np.zeros((X.shape[0],1))
        for i in range(X.shape[0]) :
            if X['Sex'].iloc[i] == 1 :
                pred[i] = 0
            else :
                pred[i] = 1
                
        return pred
    
import pandas as pd
from sklearn.preprocessing import LabelEncoder, binarize

# Null 처리 함수
def fillna(df):
    
    df['Age'].fillna(df['Age'].mean(),inplace =True)
    df['Cabin'].fillna('N',inplace=True)
    df['Embarked'].fillna('N',inplace=True)
    df['Fare'].fillna(0,inplace=True)
    return df;

# 머신러닝 알고리즘에 불필요한 속성 제거
def drop_features(df):
    df.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)
    return df

# 레이블 인코딩 수행(선실위치,성)
def format_features(df):
    df['Cabin'] = df['Cabin'].str[:1]
    features = ['Cabin','Sex','Embarked']
    for feature in features:
        le = LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])
        
    return df

# 앞에서 설정한 Data Preprocessing 함수 호출
def transform_features(df):
    df = fillna(df)
    df = drop_features(df)
    df = format_features(df)
    return df

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 원본 데이터를 재로딩, 데이터 가공, 학습데이터/테스트 데이터 분할
titanic_df = pd.read_csv('./titanic_train.csv')
y_titanic_df = titanic_df['Survived']
X_titanic_df = titanic_df.drop('Survived',axis=1)
X_titanic_df = transform_features(X_titanic_df)
X_train, X_test, y_train, y_test = train_test_split(X_titanic_df,y_titanic_df,
                                                    test_size=0.2,random_state=0)
# 위에서 생성한 Dummy Classifier를 이용하여 학습/예측/평가 수행
myclf = MyDummyClassifier()
myclf.fit(X_train,y_train)

mypredictions = myclf.predict(X_test)
#print('Dummy Classifier의 정확도는: {0:.4f}'.format(accuracy_score(y_test,mypredictions)))

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

class MyFakeClassifier(BaseEstimator):
    def fit(self,X,y):
        pass
    
    def predict(self,X):
        return np.zeros((len(X),1),dtype=bool)
    
#사이킷런의 내장 데이터인 load_digits()를 이용하여 MNIST 데이터 로딩
digits = load_digits() # MNIST는 손으로 쓴 숫자로 이루어진 대형 데이터 베이스이다.

# print(digits.data)
# print("### digits.data.shape",digits.data.shape)
# print(digits.target)
# print("### digits.target.shape:",digits.target.shape)

#digits번호가 7번이면 True이고 이를 astype(int)로 1로 변환, 7번이 아니면 False이고 0으로 변환
digits.target == 7
y = (digits.target == 7).astype(int)
X_train,X_test,y_train,y_test = train_test_split(digits.data,y,random_state=11)

# #불균형한 레이블 데이터 분포도 확인
# print("레이블 테스트 세트 크기 : ",y_test.shape)
# print("테스트 세트 레이블 0과 1의 분포도")
# print(pd.Series(y_test).value_counts())

# Dummy Classifier로 학습/예측/정확도 평가
fakeclf = MyFakeClassifier()

fakeclf.fit(X_train,y_train)
fakepred = fakeclf.predict(X_test)
#print("모든 예측을 0으로 하여도 정확도는:{:.3f}".format(accuracy_score(y_test, fakepred)))

from sklearn.metrics import accuracy_score, precision_score, recall_score

#print("정밀도:",precision_score(y_test, fakepred))
#print("재현율:",recall_score(y_test, fakepred))

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

def get_clf_eval(y_test, pred): # 평가지표 출력 함수 설정
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    print("오차 행렬")
    print(confusion)
    print("정확도: {0:.4f}, 정밀도: {1:.4f}, 재현률: {2:.4f}".format(accuracy,precision,recall))
    
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#원본 데이터를 재로딩, 데이터 가공, 학습데이터/테스트 데이터 분할
titanic_df = pd.read_csv('./titanic_train.csv')
y_titanic_df = titanic_df["Survived"]
X_titanic_df = titanic_df.drop("Survived",axis=1)
X_titanic_df = transform_features(X_titanic_df)

X_train,X_test,y_train,y_test = train_test_split(X_titanic_df, y_titanic_df,
                                                 test_size=0.20,random_state=11)

lr_clf = LogisticRegression()
lr_clf.fit(X_train,y_train)
pred = lr_clf.predict(X_test)
#get_clf_eval(y_test,pred)

pred_proba = lr_clf.predict_proba(X_test)
pred = lr_clf.predict(X_test)
#print('pred_proba()의 결과 Shape : {0}'.format(pred_proba.shape))
#print('pred_proba array에서 앞 3개만 샘플로 추출 \n:',pred_proba[:3])

#예측 확률 array와 예측 결과값 array를 concatenate하여 확률과 결과값을 한눈에 확인
pred_proba_result = np.concatenate([pred_proba,pred.reshape(-1,1)],axis=1)
#print('두개의 class 중에서 더 큰 확률을 클래스 값으로 예측 \n',pred_proba_result[:3])

from sklearn.preprocessing import Binarizer

X = [[1,-1,2],
     [2,0,0],
     [0,1.1,1.2]]

#threshold 기준값보다 같거나 작으면 0, 크면 1을 반환

binarizer = Binarizer(threshold=1.1)
#print(binarizer.fit_transform(X))

from sklearn.preprocessing import Binarizer

#threshold 기준값 설정(0.5)
custom_threshold = 0.5

pred_proba_1 = pred_proba[:,1].reshape(-1,1)

binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_1) # 모델 학습용 fit()
custom_predict=binarizer.transform(pred_proba_1)

#fit과 fit_transform의 차이점 여기서 말하는 fit은 모델을 학습시킬 때 사용하는 것이 아니다. fit() -> predict()가 아니다.
#여기서 말하는 fit()은 데이터를 전처리 할 때 사용하는 것이다. 즉, 학습데이터 세트에서 변환을 위한 기반을 설정하는 단계
#transform()은 fit()메서드에서 저장한 설정값들을 기반으로 데이터를 변환하는 메서드이다.
#fit_transform()은 fit()과 transfrom()메서드의 동작을 일련의 연속적으로 수행하기 위한 메서드이다.
#만약 train데이터로 전처리를 할 시 파이프 라인을 통해 fit_transform을 하게 되는데 이를 통해 StandardScaler()등에는 훈련 데이터 변환을 위한 여러 설정값들이 내장되어있다.
#하지만 test세트의 경우 훈련세트에서 학습한 내용을 바탕으로 적용을 해야 하기 때문에 fit_transform(),fit()을 사용하면 기존에 저장했던 설정값들이 변경된다.
#따라서 transform()메서드만 사용해야 한다. (train->fit_transform(),fit(), test->transform())


#get_clf_eval(y_test,custom_predict)

#threshold=0.4로 설정
custom_threshold = 0.4
pred_proba_1 = pred_proba[:,1].reshape(-1,1)
binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_1)
custom_predict=binarizer.transform(pred_proba_1)

#get_clf_eval(y_test,custom_predict)

#여러개의 threshold를 기반으로 한 Binarizer 예측값 계산
thresholds = [0.4,0.45,0.5,0.55,0.6]

def get_eval_by_threshold(y_test,pred_proba_c1,thresholds):
    # thresholds list객체내의 값을 차례로 iteration하면서 Evaluation 수행.
    for custom_threshold in thresholds:
        binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_c1)
        custom_predict = binarizer.transform(pred_proba_c1)
        print("임곗값:",custom_threshold)
        get_clf_eval(y_test,custom_predict)
#get_eval_by_threshold(y_test,pred_proba[:,1].reshape(-1,1),thresholds)


#precision_recall_curve를 이용하여 값 추출
from sklearn.metrics import precision_recall_curve

#레이블이 1일 때의 예측 확률을 추출
pred_proba_Class1 = lr_clf.predict_proba(X_test)[:,1]

#실제값 데이터 셋과 레이블 값이 1일 때의 예측 확률을 precision_recall_curve 인자로 입력
precisions,recalls,thresholds = precision_recall_curve(y_test,pred_proba_Class1)
# print("반환된 분류 결정 임곗값 배열의 shape: ",thresholds.shape)
# print("반환된 precisions 배열의 shape: ",precisions.shape)
# print("반환된 recals 배열의 shape: ",recalls.shape)

# print("threholds 5 sample: ",thresholds[:5])
# print("precisions 5 sample: ",precisions[:5])
# print("recalls 5 sample:",recalls[:5])

# #반환된 임계값 배열 row가 147건이므로 샘플로 10건만 추출하되, 임계값을 15 step으로 추출
# thr_index = np.arange(0,thresholds.shape[0],15)
# print("샘플 추출을 위한 임계값 배열의 index 10개 :",thr_index)
# print("샘플용 10개의 임곗값 :"),np.round(thresholds[thr_index], 2)

# #15 step 단위로 추출된 임계값에 따른 정밀도와 재현율 값
# print("샘플 임계값별 정밀도: ",np.round(precisions[thr_index],3))
# print("샘플 임계값별 재현율 :",np.round(recalls[thr_index],3))

#그래프로 출력
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


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
    
#precision_recall_curve_plot(y_test,lr_clf.predict_proba(X_test)[:,1])

#F1 Score
from sklearn.metrics import f1_score
f1 = f1_score(y_test,pred)
#print("F1 score: {0:.4f}".format(f1))

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
    print("정확도: {0:.4f}, 정밀도: {1:.4f}, 재현률: {2:.4f}, F1: {3:.4f}".format(accuracy,precision,recall,f1))

thresholds = [0.4,0.45,0.5,0.55,0.6]
pred_proba = lr_clf.predict_proba(X_test)
#get_eval_by_threshold(y_test,pred_proba[:,1].reshape(-1,1),thresholds)

from sklearn.metrics import roc_curve

# 레이블 값이 1일 때의 예측 확률을 추출
pred_proba_class1 = lr_clf.predict_proba(X_test)[:,1]

fprs, tprs, thresholds = roc_curve(y_test,pred_proba_class1)
#반환된 임곗값 배열에서 샘플로 데이터를 추출하되, 임곗값을 5 step으로 추출
#thresholds[0]은 max(예측확률)+1로 임의 설정됨. 이를 제외하기 위해 np.arange는 1부터 시작
thr_index = np.arange(1,thresholds.shape[0],5)
# print("샘플 추출을 위한 임곗값 배열의 index: ",thr_index)
# print("샘플 index로 추출한 임곗값: ",np.round(thresholds[thr_index],2))

# #5 step 단위로 추출된 임계값에 따른 FPR, TPR 값
# print("샘플 임곗값별 FPR: ",np.round(fprs[thr_index],3))
# print("샘플 임곗값별 TPR: ",np.round(tprs[thr_index],3))

def roc_curve_plot(y_test,pred_proba_c1):
    #임곗값에 따른 FPR, TPR 값을 반환받음
    fprs,tprs,thresholds = roc_curve(y_test,pred_proba_c1)
    
    #ROC Curve를 plot 곡선으로 그림
    plt.plot(fprs, tprs, label="ROC")
    #가운데 대각선 직선을 그림
    plt.plot([0,1],[0,1],'k--',label='Random')
    
    #FPR X축의 Scale을 0.1단위로 변경, X,Y 축명 설정등
    start,end = plt.xlim()
    plt.xticks(np.round(np.arange(start,end,0.1),2))
    plt.xlim(0,1);plt.ylim(0,1)
    plt.xlabel("FPR( 1- Sensitivity )"); plt.ylabel("TPR( Recall )")
    plt.legend()
    plt.show()
    
#roc_curve_plot(y_test,lr_clf.predict_proba(X_test)[:,1])

from sklearn.metrics import roc_auc_score

pred_proba = lr_clf.predict_proba(X_test)[:,1]
roc_score = roc_auc_score(y_test,pred_proba)
print("ROC AUC 값: {0:.4f}".format(roc_score))

def get_clf_eval(y_test,pred=None,pred_proba=None):
    confusion = confusion_matrix(y_test,pred)
    accuracy = accuracy_score(y_test,pred)
    precision = precision_score(y_test,pred)
    recall = recall_score(y_test,pred)
    f1 = f1_score(y_test,pred)
    
    #ROC-AUC 추가
    roc_auc = roc_auc_score(y_test,pred_proba)
    print("오차행렬")
    print(confusion)
    #ROC-AUC print 추가
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f},\
          F1: {3:.4f}, AUC: {4:.4f}'.format(accuracy,precision,recall,f1,roc_auc))
get_eval_by_threshold(y_test,pred_proba[:,1].reshape(-1,1),thresholds)
    
