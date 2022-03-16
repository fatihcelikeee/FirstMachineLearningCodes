#Data from https://www.bilkav.com/makine-ogrenmesi-egitimi/

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


tenisDatas = pd.read_csv('odev_tenis.csv')
le = preprocessing.LabelEncoder()
ohe = preprocessing.OneHotEncoder()
regressor = LinearRegression()

outlook=tenisDatas[["outlook"]].values
outlook[:,0] = le.fit_transform(tenisDatas.iloc[:,0])
outlook = ohe.fit_transform(outlook).toarray()
outlook = pd.DataFrame(data=outlook, index = range(len(outlook)), columns = ['overcast','rainy','sunny'])

windy=tenisDatas[["windy"]].values
windy[:,0] = le.fit_transform(tenisDatas.iloc[:,-2])
windy = ohe.fit_transform(windy).toarray()
windy = pd.DataFrame(data=windy, index = range(len(windy)), columns = ['False','windy'])[["windy"]]

temperatureandhumidity=tenisDatas.iloc[:,1:3]
inputs=pd.concat([outlook,temperatureandhumidity,windy], axis=1)

play=tenisDatas[["play"]].values
play[:,0] = le.fit_transform(tenisDatas.iloc[:,-1])
play = ohe.fit_transform(play).toarray()
play = pd.DataFrame(data=play, index = range(len(play)), columns = ['False','play'])[["play"]]

be=np.append(arr = np.ones((len(play),1)).astype(int), values=inputs, axis=1)

beList = inputs.iloc[:,:].values
beList = np.array(beList,dtype=float)
model = sm.OLS(play,beList).fit()
#print(model.summary())

x_train, x_test,y_train,y_test = train_test_split(inputs.iloc[:,[0,1,2,5]],play,test_size=0.33, random_state=0)
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)
#print(y_pred)
#print(y_test)
plt.plot(y_pred,y_test,"go")
#plt.show()

#humidity predict
datas=pd.concat([inputs,play],axis=1)
humidity=datas.iloc[:,4]
right=datas.iloc[:,:4]
left=datas.iloc[:,5:]
all=pd.concat([right,left],axis=1)

be=np.append(arr = np.ones((14,1)).astype(int), values=all, axis=1)
beList = all.values
beList = np.array(beList,dtype=float)
model = sm.OLS(humidity,beList).fit()
#print(model.summary())

final=pd.concat([all.iloc[:,:4],all.iloc[:,5:]], axis=1)
print(final)

x_train, x_test,y_train,y_test = train_test_split(final,humidity,test_size=0.33, random_state=0)
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
plt.plot(y_pred,y_test,"go")
plt.show()
