from statistics import LinearRegression
from tkinter import Y
from sklearn.preprocessing import PolynomialFeatures
import numpy as np 

#다항식으로 변환한 단항식 생성,[[0,1],[2,3]]의 2*2행렬 생성
X = np.arange(4).reshape(2,2)
print('일차 단항식 계수 피처:\n',X)

#degree=2인 2차 다항식으로 변환
poly = PolynomialFeatures(degree=2)
poly.fit(X)
poly_ftr = poly.transform(X)
print('변환된 2차 다항식 계수 피처:\n',poly_ftr)

def polynomial_func(X):
    y = 1 + 2 * X[:,0] + 3 * X[:,0]**2 + 4 * X[:,1]**3
    return Y

X = np.arange(4).reshape(2,2)
print('일차 단항식 계수 feature: \n',X)
y = polynomial_func(X)
print('삼차 다항식 결정값: \n',y)

#3차 다항식 변환 
poly_ftr = PolynomialFeatures(degree=3).fit_transform(X)
print('3차 다항식 계수 feature: \n',poly_ftr)

#Linear Regression에 3차 다항식 계수 feature와 3차 다항식 결정값으로 학습 후 회귀 계수 확인
model = LinearRegression()
model.fit(poly_ftr,y)
print('Polynomial 회귀 계수\n',np.round(model.coef_,2))
print('Polynomial 회귀 shape : ',model.coef_.shape)

#피처 변환과 선형회귀 적용을 각각 별도로 하는 것보다는 사이킷런의 pipeline객체를 이용해 한 번에 다항회귀를 구현
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
