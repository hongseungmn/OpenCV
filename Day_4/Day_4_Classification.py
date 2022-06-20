from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

#plt.title("3 class values with 2 Features Sample data creation")

#2차원 시각화를 위해서 피처는 2개, 클래스는 3가지 유형의 분류 샘플 데이터 생성
X_features, y_labels = make_classification(n_features=2,n_redundant=0,n_informative=2,
                                         n_classes=3,n_clusters_per_class=1,random_state=0)

#그래프 형태로 2개의 피처로 2차원 좌표 시각화, 각 클래스 값은 다른 색깔로 표시됨
#plt.scatter(X_features[:,0],X_features[:,1],marker='o',c=y_labels,s=25,edgecolors='k')
#plt.show()

import numpy as np

# Classifier의 Decision Boundary를 시각화 하는 함수
def visualize_boundary(model, X, y):
    fig,ax = plt.subplots()
    
    # 학습 데이타 scatter plot으로 나타내기
    ax.scatter(X[:, 0], X[:, 1], c=y, s=25, cmap='rainbow', edgecolor='k',
               clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim_start , xlim_end = ax.get_xlim()
    ylim_start , ylim_end = ax.get_ylim()
    
    # 호출 파라미터로 들어온 training 데이타로 model 학습 . 
    model.fit(X, y)
    # meshgrid 형태인 모든 좌표값으로 예측 수행. 
    xx, yy = np.meshgrid(np.linspace(xlim_start,xlim_end, num=200),np.linspace(ylim_start,ylim_end, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    # contourf() 를 이용하여 class boundary 를 visualization 수행. 
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap='rainbow', clim=(y.min(), y.max()),
                           zorder=1)
    plt.show()
   
from sklearn.tree import DecisionTreeClassifier

#특정한 트리 생성 제약 없는 결정 트리의 학습과 결정 경계 시각화
dt_clf = DecisionTreeClassifier().fit(X_features,y_labels)
#visualize_boundary(dt_clf,X_features,y_labels)

dt_clf = DecisionTreeClassifier(min_samples_leaf=6).fit(X_features,y_labels)
#visualize_boundary(dt_clf,X_features,y_labels)

import pandas as pd
import matplotlib.pyplot as plt

#features.txt 파일에는 피처 이름 index와 피처명이 공백으로 분리되어 있음. 이를 DataFrame으로 로드.
feature_name_df = pd.read_csv('./human_activity/features.txt',sep='\s+',header=None,
                              names=["columns_index","columns_name"])
#print(feature_name_df)
#피처명 index를 제거하고, 피처명만 리스트 객체로 생성한 뒤 샘플로 10개만 추출
feature_name = feature_name_df.iloc[:,1].values.tolist() # tolist() 같은 위치의 데이터끼리 묶어준다.
# print("전체 피처명에서 10개만 추출: ",feature_name[:10])

feature_dup_df = feature_name_df.groupby('columns_name').count()
# print(feature_dup_df[feature_dup_df['columns_index']>1].count())
# print(feature_dup_df[feature_dup_df['columns_index']>1].head())

def get_new_feature_name_df(old_feafure_name_df):
    feature_dup_df = pd.DataFrame(data=old_feafure_name_df.groupby('column_name').cumcount(),columns=['dup_cnt'])
    feature_dup_df = feature_dup_df.reset_index() # 인덱스를 초기화
    new_feature_name_df = pd.merge(old_feafure_name_df.reset_index(),feature_dup_df,how='outer')
    new_feature_name_df['column_name'] = new_feature_name_df[['column_name','dup_cnt']].apply(lambda x : x[0]+'_'+str(x[1])
                                                                                              if x[1] >0 else x[0], axis=1)
    new_feature_name_df = new_feature_name_df.drop(['index'],axis=1)
    return new_feature_name_df

import pandas as pd 
def get_human_dataset():
    
    #각 파일은 공백으로 분리되어 있으므로 read_csv에서 공백 문자를 sep으로 할당
    feature_name_df = pd.read_csv('./human_activity/features.txt',sep='\s+',
                                  header=None,names=['column_index','column_name'])
    #중복된 피처명을 수정하는 get_new_feature_name_df()를 이용, 신규 피처명 DataFrame 생성
    new_feature_name_df = get_new_feature_name_df(feature_name_df)
    
    #DataFrame에 피처명을 칼럼으로 부여하기 위해 리스트 객체로 다시 변환
    feature_name = new_feature_name_df.iloc[:,1].values.tolist()
    
    #학습 피처 데이터세트와 테스트 피처 데이터를 DataFrame으로 로딩. 칼럼명은 feature_name 적용
    X_train = pd.read_csv('./human_activity/train/X_train.txt',sep='\s+',names=feature_name)
    X_test = pd.read_csv('./human_activity/test/X_test.txt',sep='\s+',names=feature_name)
    
    #학습 레이블과 테스트 레이블 데이터를 DataFrame으로 로딩하고 칼럼명은 action으로 부여
    y_train = pd.read_csv('./human_activity/train/y_train.txt',sep='\s+',header=None,names=['action'])
    y_test = pd.read_csv('./human_activity/test/y_test.txt',sep='\s+',header=None,names=['action'])
    
    #로드된 학습/테스트용 DataFrame을 모두 반환
    return X_train,X_test,y_train,y_test

X_train,X_test,y_train,y_test = get_human_dataset()

# print('##학습 피처 데이터 셋 info()##')
# print(X_train.info())
# print(y_train['action'].value_counts())

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#예제 반복 시마다 동일한 예측 결과 도출을 위해 random_state 설정
dt_clf = DecisionTreeClassifier(random_state=156)
dt_clf.fit(X_train,y_train)
pred = dt_clf.predict(X_test)
accuracy = accuracy_score(y_test,pred)
#print('결정 트리 예측 정확도:{0:.4f}'.format(accuracy))

#DecisionTreeClassifier의 하이퍼 파라미터 추출
#print('DecisionTreeClassifier 기본 하이퍼 파라미터 \n',dt_clf.get_params())

from sklearn.model_selection import GridSearchCV 
#하이퍼파라미터를 순차적으로 입력해 학습을 하고 측정을 하면서 가장 좋은 파라미터를 알려준다 grid 파라미터 안에서 집합을 만들고 적용하면 최적화된 파라미터를 뽑아낼 수 있다.
params = {'max_depth':[6,8,10,12,16,20,24]}

grid_cv = GridSearchCV(dt_clf,param_grid=params, scoring='accuracy',cv=5,verbose=1)
grid_cv.fit(X_train,y_train)
#print('GridSearchCV 최고 평균 정확도 수치:{0:.4f}'.format(grid_cv.best_score_))
#print('GridSearchCV 최적 하이퍼 파라미터:',grid_cv.best_params_)

#GridSearchCV 객체의 cv_results_속성을 DataFrame으로 생성
cv_results_df = pd.DataFrame(grid_cv.cv_results_)

#max_depth 파라미터 값과 그때의 테스트 세트, 학습 데이터 세트의 정확도 수치 추출
cv_results_df[['param_max_depth','mean_test_score']]


max_depths = [6,8,10,12,16,20,24]
#max_depth값을 변화시키면서 그때마다 학습과 테스트 세트에서의 예측 성능 측정
# for depth in max_depths:
#     dt_clf = DecisionTreeClassifier(max_depth=depth,random_state=156)
#     dt_clf.fit(X_train,y_train)
#     pred = dt_clf.predict(X_test)
#     accuracy = accuracy_score(y_test,pred)
#     print('max_depth = {0} 정확도: {1:4f}'.format(depth,accuracy))
    
params = {'max_depth':[8,12,16,20],
          'min_samples_split':[16,24]}

grid_cv = GridSearchCV(dt_clf,param_grid=params,scoring='accuracy',cv=5,verbose=1)
grid_cv.fit(X_train,y_train)
# print('GridSearchCV 최고 평균 정확도 수치: {0:.4f}'.format(grid_cv.best_score_))
# print('GridSearchCV 최적 하이퍼 파라미터: ',grid_cv.best_params_)

best_df_clf = grid_cv.best_estimator_
pred1 = best_df_clf.predict(X_test)
accuracy = accuracy_score(y_test,pred1)
# print('결정 트리 예측 정확도:{0:.4f}'.format(accuracy))

import seaborn as sns

ftr_importances_values = best_df_clf.feature_importances_
#Top 중요도로 정렬을 쉽게 하고, 시본(Seaborn)의 막대그래프로 쉽게 표현하기 위해 Series 변환
ftr_importances = pd.Series(ftr_importances_values,index=X_train.columns)
#중요도 값 순으로 Series 정렬
ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]
# plt.figure(figsize=(8,6))
# plt.title('Feature importances Top 20')
# sns.barplot(x=ftr_top20,y=ftr_top20.index)
# plt.show()

#결정 트리에서 사용한 get_human_dataset()를 이용해 학습/테스트용 DataFrame 반환
X_train,X_test,y_train,y_test = get_human_dataset()

#랜덤 포레스트 학습 및 별도의 테스트 세트로 예측 성능 평가
rf_clf = RandomForestClassifier(random_state=0)
rf_clf.fit(X_train,y_train)
pred = rf_clf.predcit(X_test)
accuracy = accuracy_score(y_test,pred)
#print('랜덤 포레스트 정확도: {0:.4f}'.format(accuracy))

from sklearn.ensemble import GradientBoostingClassifier
import time
import warnings
warnings.filterwarnings('ignore')

X_train,X_test,y_train,y_test = get_human_dataset()

#GBM 수행 시간 측정을 위함. 시작 시간 설정
start_time = time.time()
gb_clf = GradientBoostingClassifier(random_state=0)
gb_clf.fit(X_train,y_train)
gb_pred = gb_clf.predict(X_test)
gb_accuracy = accuracy_score(y_test,gb_pred)

# print('GBM 정확도:{0:.4f}'.fotmat(gb_accuracy))
# print('GBM 수행 시간:{0:.1f}'.fotmat(time.time() - start_time))

params = {
    'n_estimators' : [100,500],
    'learning_rate' : [0.05,0.1]
}

grid_cv = GridSearchCV(gb_clf,param_grid=params,cv=2,verbose=1) # cv는 교차검증 세트
grid_cv.fit(X_train,y_train)
#print('최적 하이퍼 파라미터 :\n',grid_cv.best_params_)
#print('최고 예측 정확도 :{0.4f}'.format(grid_cv.best_score_))

#GridSearchCV를 이용해 최적으로 학습된 estimator로 예측 수행
gb_pred = grid_cv.best_estimator_.predict(X_test)
gb_accuracy = accuracy_score(y_test,gb_pred)
#print('GBM 정확도 : {0.4f}'.format(gb_accuracy))