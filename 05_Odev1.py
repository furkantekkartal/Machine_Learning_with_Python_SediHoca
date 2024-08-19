# Kutuphanelerin yuklenmesi

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Veri yukleme

data = pd.read_csv('odev_tenis.csv')

# Encoder: Non-numerik veriler(Kategoruk->Numeric)
from sklearn import preprocessing
le = preprocessing.LabelEncoder() 
ohe = preprocessing.OneHotEncoder() 

## Encoder: (outlook)
outlook  = data.iloc[:,0:1].values 
outlook[:,0]= le.fit_transform(data.iloc[:,0])  
outlook = ohe.fit_transform(outlook).toarray()  

## Encoder: (windy) 
windy = data.iloc[:, -2:-1].astype(bool).values # Original values ('T' and 'F')
#windy[:, 0] = le.fit_transform(data.iloc[:, 3])  # transformed values ('FALSE' = 0 and 'TRUE' = 1)


## Encoder: (play) 
play  = data.iloc[:,-1:].values # Original values ('y' and 'n')
play[:,0]= le.fit_transform(data.iloc[:,-1]) # transformed values ('yy' = 0 and 'n' = 1)


# DataFrame Prepearing

df_outlook = pd.DataFrame(data=outlook, index=range(14),columns=['overcast','rainy', 'sunny'])
df_windy   = pd.DataFrame(data=windy,   index=range(14),columns=['windy?'])
df_play    = pd.DataFrame(data=play,    index=range(14),columns=['play?'])
df_temp    = pd.DataFrame(data=data.iloc[:,1:2],    index=range(14),columns=['temperature'])

df_variables = pd.concat([df_outlook,df_play,df_temp,df_windy],axis=1)
df_y = pd.DataFrame(data=data.iloc[:,2:3],    index=range(14),columns=['humidity'])

# Data split for test and train

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df_variables, df_y, test_size=0.33,random_state=0)

# Tahmin humidity

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

pred_y = regressor.predict(x_test)


# Backward elimination

import statsmodels.api as sm

X_1 = np.append(arr= np.ones((14,1)).astype(int), values = df_variables, axis = 1)

# Step 1: let's say Significance Level(SL) = 0.01

# Step 2: Include all varaiables and build a model
X_1 = df_variables.iloc[:,[0,1,2,3,4,5]].values
X_1 = np.array(X_1, dtype =  float)
model = sm.OLS(df_y, X_1).fit()
print(model.summary()) 

X_1 = df_variables.iloc[:,[0,1,2,3,4]].values
X_1 = np.array(X_1, dtype =  float)
model = sm.OLS(df_y, X_1).fit()
print(model.summary()) 

X_1 = df_variables.iloc[:,[0,1,3,4]].values
X_1 = np.array(X_1, dtype =  float)
model = sm.OLS(df_y, X_1).fit()
print(model.summary()) 

X_1 = df_variables.iloc[:,[1,3,4]].values
X_1 = np.array(X_1, dtype =  float)
model = sm.OLS(df_y, X_1).fit()
print(model.summary()) 

X_1 = df_variables.iloc[:,[1,4]].values
X_1 = np.array(X_1, dtype =  float)
model = sm.OLS(df_y, X_1).fit()
print(model.summary()) 

X_1 = df_variables.iloc[:,[4]].values
X_1 = np.array(X_1, dtype =  float)
model = sm.OLS(df_y, X_1).fit()
print(model.summary()) 


# Tahmin humidity2

x_train2, x_test2, y_train2, y_test2 = train_test_split(df_temp, df_y, test_size=0.33,random_state=0)

r2 = LinearRegression()
r2.fit(x_train2, y_train2)

pred_y2 = r2.predict(x_test2)
