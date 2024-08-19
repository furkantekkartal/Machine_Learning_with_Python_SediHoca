# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 21:05:18 2024

@author: furka
"""

# Kutuphanelerin yuklenmesi

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Veri yukleme

veriler = pd.read_csv('veriler.csv')
boykilo = veriler[['boy','kilo']]
Yas = veriler.iloc[:,1:4].values 


# Encoder: (Ulke) Non-numerik veriler(Kategoruk->Numeric)

from sklearn import preprocessing

ulke  = veriler.iloc[:,0:1].values 

le = preprocessing.LabelEncoder() 
ulke[:,0]= le.fit_transform(veriler.iloc[:,0])  

ohe = preprocessing.OneHotEncoder() 
ulke = ohe.fit_transform(ulke).toarray()  


# Encoder: (cinsiyet) 

c  = veriler.iloc[:,-1:].values # Original values ('e' and 'k')
c[:,-1]= le.fit_transform(veriler.iloc[:,-1]) # transformed values ('e' = 0 and 'k' = 1)
c = ohe.fit_transform(c).toarray()  # 2 column format ('e' = 1,0 and 'k' = 0,1)

# Numpy dizileri DataFrame donusumu

sonuc = pd.DataFrame(data=ulke, index=range(22), columns=['fr','tr','us'])
  # Only converted non-numeric columns

sonuc2 = pd.DataFrame(data=Yas, index=range(22), columns=['boy', 'kilo', 'yas']) #We said only 'Yas' but dataframe has 3 column.
  # Only numeric columns

cinsiyet = veriler.iloc[:,-1].values
  # Take the target column from original DataFrame

sonuc3 = pd.DataFrame(data=c[:,:1], index=range(22),columns=['cinsiyet(e?)'])
  # Only target column

s=pd.concat([sonuc,sonuc2],axis=1) #axis1 ayni index nolarini birlestir demek
  # Join non-numeric + numeric columns

s2=pd.concat([s,sonuc3],axis=1)
  # Join non-numeric + numeric columns + target column


# Data split for test and train

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(s, sonuc3, test_size=0.33,random_state=0)
  #double click the these 4 dataframes in Variable Explorer

# Tahmin cinsiyet

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

cisiyet_pred = regressor.predict(x_test)

# Tahmin boy

boy = s2.iloc[:,3:4].values # bagimli degisken

sol = s2.iloc [:,:3]  # boy un sol tarafi
sag = s2.iloc [:, 4:] # boy un sag tarafi
veri = pd.concat([sol, sag], axis=1) # bagimsiz degiskenlerin tumu

x_train, x_test, y_train, y_test = train_test_split(veri, boy, test_size=0.33,random_state=0)


r2 = LinearRegression()
r2.fit(x_train, y_train)

boy_pred = r2.predict(x_test)






































