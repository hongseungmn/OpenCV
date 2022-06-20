import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.datasets import load_boston

# boston 데이터 로드
boston = load_boston()

# boston 데이터 세트 DataFrame 변환
boston_df = pd.DataFrame(boston.data,columns= boston.feature_names)

#boston 데이터 세트의 target 배열은 주택 가격임. 이를 PRICE 칼럼으로 DataFrame에 추가함
boston_df['PRICE'] = boston.target
# print('Boston 데이터 세트 크기: ',boston_df.shape)
# print(boston_df.head(5))

#2개의 행과 4개의 열을 가진 subplots를 이용. axs는 4*2개의 ax를 가짐
fig,axs = plt.subplots(figsize=(16,8),ncols=4,nrows=2)
lm_features = ['RM','ZN','INDUS','NOX','AGE','PTRATIO','LSTAT','RAD']
# for i,feature in enumerate(lm_features):
#     row = int(i/4)
#     col = i%4
#     sns.regplot(x=feature,y="PRICE",data=boston_df,ax=axs[row][col])
# plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

y_target = boston_df['PRICE']
X_data = boston_df.drop(['PRICE'],axis=1,inplace=False)

X_train,X_test,y_train,y_test = train_test_split(X_data,y_target,test_size=0.3,random_state=156)

#선형회귀 OLS로 학습/예측/평가 수행
lr = LinearRegression()
lr.fit(X_train,y_train)

y_preds = lr.predict(X_test)
mse = mean_squared_error(y_test,y_preds)
rmse = np.sqrt(mse)

# print('MSE:{0:.3f}, RMSE:{1:.3f}'.format(mse,rmse))
# print('Varience score :{0:3f}'.format(r2_score(y_preds,y_test)))

# print('절편 값:',lr.intercept_)
# print('회귀 계수 값:',np.round(lr.coef_,1))

coeff = pd.Series(data=np.round(lr.coef_,1),index=X_data.columns)
print(coeff.sort_values(ascending=True))


from sklearn.model_selection import cross_val_score

#cross_val_score()로 5 폴드 세트로 MSE를 구한 뒤 이를 기반으로 다시 RMSE 구함  
neg_mse_scores = cross_val_score(lr,X_data,y_target,scoring="neg_mean_squared_error",cv=5)
rmse_score = np.sqrt(-1*neg_mse_scores)
avg_rmse = np.mean(rmse_score)

#cross_val_score(scoring="neg_mean_squared_error")로 반환된 값은 모두 음수
# print('5 folds의 개별 Negative MSE scores : ',np.round(neg_mse_scores,2))
# print('5 folds의 개별 RMSE scores : ',np.round(rmse_score,2))
# print('5 folds의 평균 RMSE {0:.3f}: '.format(avg_rmse))