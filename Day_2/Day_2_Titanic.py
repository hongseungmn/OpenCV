#%%
import matplotlib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns



titanic_df = pd.read_csv('./titanic_train.csv')
'''
print(titanic_df.head(3))
print('\n ### train 데이터 정보 ###  \n')
print(titanic_df.info())
'''
#나이의 결측값은 평균으로 채움
titanic_df['Age'].fillna(titanic_df['Age'].mean(),inplace = True)

#Cabin과 Embarkedsms N으로 채움                  
titanic_df['Cabin'].fillna('N',inplace=True)
titanic_df['Embarked'].fillna('N',inplace=True)

#print('데이터 세트 Null 값 갯수 ',titanic_df.isnull().sum().sum())
'''
print(' Sex 값 분포 : \n',titanic_df['Sex'].value_counts())
print(' Cabin 값 분포 : \n',titanic_df['Cabin'].value_counts())
print(' Embarked 값 분포 : \n',titanic_df['Embarked'].value_counts())
'''

#값을 대표문자 1개로 축소
titanic_df['Cabin'] = titanic_df['Cabin'].str[:1]
#print(titanic_df['Cabin'].head(3))

#성별 생존자 수를 집계 
titanic_df.groupby(['Sex','Survived'])['Survived'].count()
#print(titanic_df.head(3))

sns.barplot(x='Sex', y = 'Survived',data= titanic_df)

sns.barplot(x='Pclass', y='Survived',hue='Sex',data=titanic_df)

def get_category(age):
    cat =''
    if age <= -1 : cat = 'Unknown'
    elif age <= 5 : cat = 'Baby'
    elif age <= 12 : cat = 'Child'
    elif age <= 18 : cat = 'Teenager'
    elif age <= 25 : cat = 'Student'
    elif age <= 35 : cat = 'Young Adult'
    elif age <= 60 : cat = 'Adult'
    else : cat = 'Elderly'
    
    return cat

plt.figure(figsize=(10,6))

group_names = ['Unknown','Baby','Child','Teenager','Student','Young Adult','Adult','Elderly']

#lambda 식에 위에서 생성한 get_category()함수를 반환값으로 지정
titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x : get_category(x))
sns.barplot(x='Age_cat', y='Survived',hue = 'Sex',data = titanic_df,order=group_names)
titanic_df.drop('Age_cat',axis=1,inplace=True)

from sklearn import preprocessing

def encode_features(dataDf):
    features = ['Cabin','Sex','Embarked']
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(dataDf[feature])
        dataDf[feature] = le.transform(dataDf[feature])
    
    return dataDf

titanic_df = encode_features(titanic_df)
#print(titanic_df.head())

from sklearn.preprocessing import LabelEncoder

def fillna(df):
    df['Age'].fillna(df['Age'].mean(),inplace = True)
    df['Cabin'].fillna('N',inplace=True)
    df['Embarked'].fillna('N',inplace = True)
    return df

def drop_features(df):
    df.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)
    return df

def format_features(df):
    df['Cabin'] = df['Cabin'].str[:1]
    features = ['Cabin','Sex','Embarked']
    for feature in features:
        le = LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    return df

def transform_features(df):
    df = fillna(df)
    df = drop_features(df)
    df = format_features(df)
    return df


titanic_df = pd.read_csv('./titanic_train.csv')
y_titanic_df = titanic_df['Survived']
X_titanic_df = titanic_df.drop('Survived',axis=1)

X_titanic_df = transform_features(X_titanic_df)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X_titanic_df,y_titanic_df,test_size=0.2,random_state=11)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

tree_clf = DecisionTreeClassifier(random_state=11)
randomforest_clf = RandomForestClassifier(random_state=11)
logistic_clf = LogisticRegression()


#tree model
tree_clf.fit(X_train,y_train)
tree_pred = tree_clf.predict(X_test)
#print('accuracy of DecisionTreeClassifier is : {0:.4f}'.format(accuracy_score(y_test,tree_pred)))
#print('matrics of DecisionTreeClassifier is \n',confusion_matrix(y_test,tree_pred))


#randomforest model
randomforest_clf.fit(X_train,y_train)
randomforest_pred = randomforest_clf.predict(X_test)
#print('accuracy of RandomForestClassifier is :{0:.4f}'.format(accuracy_score(y_test,randomforest_pred)))
#print('matrics of RandomForestClassifier is \n',confusion_matrix(y_test,randomforest_pred))       


#LogisticRegression model
logistic_clf.fit(X_train,y_train)
logistic_pred = logistic_clf.predict(X_test)
#print('accuracy of LogisticRegression is :{0:.4f}'.format(accuracy_score(y_test,logistic_pred)))
#print('matrics of LogisticRegression is \n',confusion_matrix(y_test,logistic_pred))

from sklearn.model_selection import KFold

def exec_kfold(clf,folds=5):
    kfold = KFold(n_splits=folds)
    scores = []
    
    for iter_count, (train_index, test_index) in enumerate(kfold.split(X_titanic_df)):
        X_train,X_test = X_titanic_df.values[train_index], X_titanic_df.values[test_index]
        y_train,y_test = y_titanic_df.values[train_index], y_titanic_df.values[test_index]
        
        clf.fit(X_train,y_train)
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test,predictions)
        scores.append(accuracy)
        #print("교차 검증 {0} 정확도: {1:.4f}".format(iter_count,accuracy))
        
    mean_score = np.mean(scores)
#    print("평균 정확도 : {0:.4f}".format(mean_score))
exec_kfold(tree_clf,folds=5)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_clf,X_titanic_df,y_titanic_df,cv=5)
#for iter_count,accuracy in enumerate(scores):
    #print("교차 검증 {0} 정확도: {1:.4f}".format(iter_count,accuracy))
    
#print("평균 정확도: {0:.4f}".format(np.mean(scores)))

from sklearn.model_selection import GridSearchCV

parameters = {'max_depth':[2,3,5,10],
              'min_samples_split': [2,3,5],
              'min_samples_leaf':[1,5,8]}

grid_clf = GridSearchCV(tree_clf,param_grid=parameters,scoring='accuracy',cv=5)
grid_clf.fit(X_train,y_train)

print('GridSearchCV 최적 하이퍼 파라미터 :',grid_clf.best_params_)
print('GridSearchCV 최고 정확도 : {0:.4f}'.format(grid_clf.best_score_))
best_clf = grid_clf.best_estimator_

dpredicitons = best_clf.predict(X_test)
accuracy = accuracy_score(y_test,dpredicitons)
print("테스트 세트에서의 DecisionTreeClassifier 정확도 : {0:.4f}".format(accuracy))



