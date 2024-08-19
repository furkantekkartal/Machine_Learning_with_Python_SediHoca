# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 21:05:18 2024

@author: furka
"""

#Ders 6: Kutuphanelerin yuklenmesi

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#veri yukleme
veriler = pd.read_csv('eksikveriler.csv')
print (veriler)

boy = veriler[['boy']]
#print (boy)

class insan:
    boy = 180
    def kosmak(self, b):
        return b+10

ali = insan()
print (ali.boy)
print (ali.kosmak(20))

l = [1,3,4]
print (l)

# Eksik veriler

## Numerik veriler


Yas = veriler.iloc[:,1:4].values #iloc(integer location) (: tum satirlar, 1:4 1den sonra 4e kadar(4dahil degil))
# remember, in Python, slicing is exclusive of the end index. so, these columns are 'boy', 'kilo', and 'yas'.
print(Yas)

from sklearn.impute import SimpleImputer 

imputer = SimpleImputer(missing_values=np.nan, strategy='mean') #create an object frol SimpleImputer Class

imputer = imputer.fit(Yas[:,0:3]) #call fit method of imputer object. Parameter(Yas[:,1:4]): sadece sayisal veriler
# it is not Yas[:,1:4], because Yas only has 3 columns (indices 0, 1, and 2).

Yas[:,0:3] = imputer.transform(Yas[:,0:3]) #yas listesine yeni deger ata <= Transformer methodunun ciktisi(eksiksiz 3 kolon)
print(Yas)

## Non-numerik veriler

ulke  = veriler.iloc[:,0:1].values
print (ulke)

from sklearn import preprocessing

le = preprocessing.LabelEncoder() # A LabelEncoder object is created.

ulke[:,0]= le.fit_transform(veriler.iloc[:,0])
print(ulke)


ohe = preprocessing.OneHotEncoder() # A OneHotEncoder object is created.
ulke = ohe.fit_transform(ulke).toarray()
# This creates binary columns for each unique country, where 1 indicates 
#the presence of that country and 0 indicates absence.
# The result is converted to a dense array with toarray() and printed.
print (ulke)

#dataframe birlestirme

sonuc = pd.DataFrame(data=ulke, index=range(22), columns=['fr','tr','us'])
print(sonuc)

sonuc2 = pd.DataFrame(data=Yas, index=range(22), columns=['boy', 'kilo', 'yas'])
print (sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data=cinsiyet, index=range(22),columns=['cinsiyet'])
print(sonuc3)


s=pd.concat([sonuc,sonuc2],axis=1) #axis1 ayni index nolarini birlestir demek
print(s)

s2=pd.concat([s,sonuc3],axis=1) #axis1 ayni index nolarini birlestir demek
print(s2)


# data frame bolme

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(s, sonuc3, test_size=0.33,random_state=0)
#double click the those dataframes in Variable Explorer


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
#X_test = sc.fit_transform(x_test)

