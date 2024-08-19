# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 00:21:31 2024

@author: furka
"""

# Kutuphanelerin yuklenmesi

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Veri yukleme

data = pd.read_csv('maaslar.csv')

# Data frame sliceing

x = data.iloc[:,1:2]
y = data.iloc[:,-1:]

X = x.values
Y = y.values


# Linear Regression

from sklearn.linear_model import LinearRegression
lin_reg1 = LinearRegression()
lin_reg1.fit(x, y)

#plt.scatter(x, y, color = 'red')
#plt.plot(x,lin_reg1.predict(x), color = 'blue')
#plt.show()


# Polynomial Regression (Degree 2)

from sklearn.preprocessing import PolynomialFeatures
poly_reg2 = PolynomialFeatures(degree = 2)   # Create polynom object
x_poly2 = poly_reg2.fit_transform(x)          # Transform x values to polynom

lin_reg2 = LinearRegression()               # Create new linear object
lin_reg2.fit(x_poly2, y)                     # fit transformed x values to linear 

#plt.scatter(x, y, color = 'red') 
#plt.plot(x, lin_reg2.predict(x_poly2), color = 'blue')
#plt.show()

# Polynomial Regression (Degree 4)

from sklearn.preprocessing import PolynomialFeatures
poly_reg4 = PolynomialFeatures(degree = 4)   # Create polynom object
x_poly4 = poly_reg4.fit_transform(x)          # Transform x values to polynom

lin_reg4 = LinearRegression()               # Create new linear object
lin_reg4.fit(x_poly4, y)                     # fit transformed x values to linear 

#plt.scatter(x, y, color = 'red') 
#plt.plot(x, lin_reg4.predict(x_poly4), color = 'blue')
#plt.show()


# Predictions


lin_pred_1 = lin_reg1.predict(pd.DataFrame([[11]], columns=['Egitim Seviyesi']))
lin_pred_2 = lin_reg1.predict(pd.DataFrame([[6.5]], columns=['Egitim Seviyesi']))
poly_pred_1 = lin_reg4.predict(poly_reg4.fit_transform(pd.DataFrame([[11]], columns=['Egitim Seviyesi'])))
poly_pred_2 = lin_reg4.predict(poly_reg4.fit_transform(pd.DataFrame([[6.5]], columns=['Egitim Seviyesi'])))

'''
print('Linear Regression  | x = 11  | prediction = ', lin_pred_1)
print('Polynom Regression | x = 11  | prediction = ', poly_pred_1)
print('')
print('Linear Regression  | x = 6.5 | prediction = ', lin_pred_2)
print('Polynom Regression | x = 6.5 | prediction = ', poly_pred_2)
'''

#Data scaling

from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
x_scaled = sc1.fit_transform(x)

sc2 = StandardScaler()
y_scaled = sc1.fit_transform(y.values.reshape(-1, 1)).ravel()


# SVR
from sklearn.svm import SVR

svr_rbf = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
svr_lin = SVR(kernel="linear", C=100, gamma="auto")
svr_poly = SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1)

svr_lin.fit(x_scaled, y_scaled)     # Blue   = Linear
svr_poly.fit(x_scaled, y_scaled)    # Green  = Poly
svr_rbf.fit(x_scaled, y_scaled)     # Purple = Radial Basis Function (RBF)

'''
plt.scatter(x_scaled, y_scaled, color='red')
plt.plot(x_scaled, svr_lin.predict(x_scaled), color='blue')
plt.plot(x_scaled, svr_poly.predict(x_scaled), color='green')
plt.plot(x_scaled, svr_rbf.predict(x_scaled), color='purple')
plt.show()
'''

# DecisionTree

from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(x,y)

Z = X+0.5
K = X-0.6

'''
plt.scatter(x,y, color= 'red')
plt.plot(x, r_dt.predict(X),color = 'blue')
plt.plot(x, r_dt.predict(Z),color = 'green')
plt.plot(x, r_dt.predict(K),color = 'purple')
plt.show()
'''

# Random Forest

from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators = 10 ,random_state=0)
rf_reg.fit(X,Y.ravel())

#print(rf_reg.predict([[6.6]]))
'''
plt.scatter(x,y, color= 'red')
plt.plot(x, rf_reg.predict(X),color = 'blue')
plt.plot(x, rf_reg.predict(Z),color = 'green')
plt.plot(x, rf_reg.predict(K),color = 'purple')
plt.show()
'''



