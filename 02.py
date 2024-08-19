# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 21:05:18 2024

@author: furka
"""

#1. Kutuphanelerin yuklenmesi

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#2. Veri yukleme

veriler = pd.read_csv('satislar.csv')

aylar = veriler[['Aylar']]
satislar = veriler [['Satislar']]

#aylar2=veriler.iloc[:,0:1].values
#satislar2=veriler.iloc[:,1:2].values


# 5. Data split for test and train

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(aylar, satislar, test_size=0.33,random_state=0)
  #double click the these 4 dataframes in Variable Explorer

#5.1 Veri olcekleme
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

'''
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)
'''

# 6. Lineer Regresyon

from sklearn.linear_model import LinearRegression
lr=LinearRegression()

lr.fit(x_train, y_train)

tahmin = lr.predict(x_test)


x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train,y_train)
plt.plot(x_test, tahmin)
plt.title('aylara gore satis')
plt.xlabel('aylar')
plt.ylabel('satislar')






