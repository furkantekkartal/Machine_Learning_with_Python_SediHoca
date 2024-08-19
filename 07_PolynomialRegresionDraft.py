# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 22:31:42 2024

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


# Linear Regression

from sklearn.linear_model import LinearRegression
lin_reg1 = LinearRegression()
lin_reg1.fit(x, y)

plt.scatter(x, y, color = 'red')
plt.plot(x,lin_reg1.predict(x), color = 'blue')
plt.show()


# Polynomial Regression (Degree 2)

from sklearn.preprocessing import PolynomialFeatures
poly_reg2 = PolynomialFeatures(degree = 2)   # Create polynom object
x_poly2 = poly_reg2.fit_transform(x)          # Transform x values to polynom

lin_reg2 = LinearRegression()               # Create new linear object
lin_reg2.fit(x_poly2, y)                     # fit transformed x values to linear 

plt.scatter(x, y, color = 'red') 
plt.plot(x, lin_reg2.predict(x_poly2), color = 'blue')
plt.show()

# Polynomial Regression (Degree 4)

from sklearn.preprocessing import PolynomialFeatures
poly_reg4 = PolynomialFeatures(degree = 4)   # Create polynom object
x_poly4 = poly_reg4.fit_transform(x)          # Transform x values to polynom

lin_reg4 = LinearRegression()               # Create new linear object
lin_reg4.fit(x_poly4, y)                     # fit transformed x values to linear 

plt.scatter(x, y, color = 'red') 
plt.plot(x, lin_reg4.predict(x_poly4), color = 'blue')
plt.show()



# Predictions

lin_pred_1 = lin_reg1.predict([[11]])
lin_pred_2 = lin_reg1.predict([[6.5]])
poly_pred_1 = lin_reg4.predict(poly_reg4.fit_transform([[11]]))
poly_pred_2 = lin_reg4.predict(poly_reg4.fit_transform([[6.5]]))

# Plotting
plt.figure(figsize=(12, 5))

# Linear Regression Plot
plt.subplot(1, 2, 1)
plt.scatter(x, y, color='red', label='Data')
plt.plot(x, lin_reg1.predict(x), color='blue', label='Linear Regression')
plt.scatter([11, 6.5], [lin_pred_1, lin_pred_2], color='green', s=100, label='Predictions')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.title('Linear Regression')
plt.legend()

# Polynomial Regression Plot
plt.subplot(1, 2, 2)
plt.scatter(x, y, color='red', label='Data')
plt.plot(x, lin_reg4.predict(poly_reg4.fit_transform(x)), color='blue', label='Polynomial Regression')
plt.scatter([11, 6.5], [poly_pred_1, poly_pred_2], color='green', s=100, label='Predictions')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.title('Polynomial Regression')
plt.legend()

plt.tight_layout()
plt.show()

# Print prediction results
print(f"Linear Regression Predictions    : {lin_pred_1[0][0]:.2f}, {lin_pred_2[0][0]:.2f}")
print(f"Polynomial Regression Predictions: {poly_pred_1[0][0]:.2f}, {poly_pred_2[0][0]:.2f}")



'''
# PolynomialFeatures:

    This class is used to generate polynomial and interaction features.
    When we set degree = 2, it creates features for x^1 and x^2.
    If x was originally [x1], after transformation it becomes [1, x1, x1^2].
    
# Why we still use LinearRegression:
    
    The key insight is that we're not actually using a different regression algorithm. 
    We're still using Linear Regression, but with transformed features.
    After we transform our features using PolynomialFeatures, we're essentially 
 creating a linear model in a higher-dimensional space.
    The equation goes from y = b0 + b1x to y = b0 + b1x + b2*x^2.
    This is still linear in terms of the coefficients (b0, b1, b2), even though it's 
 quadratic in terms of x.
 
#The process:
    
    We transform our original features (x) into polynomial features (x_poly).
    We then use these polynomial features with a standard LinearRegression model.
    This allows the model to capture non-linear relationships in the original feature space.

# Advantages of this approach:
    We can use the well-established Linear Regression algorithm and its implementations.
    We can easily control the complexity of the model by adjusting the degree of the polynomial.
    It's computationally efficient compared to implementing a separate polynomial regression algorithm.

    In essence, polynomial regression as implemented here is a clever use of 
 feature engineering combined with linear regression, rather than a completely 
 separate algorithm. This approach allows us to model non-linear relationships 
 while still using the simple and well-understood linear regression model.
'''