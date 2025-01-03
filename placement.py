# -*- coding: utf-8 -*-
"""placement.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1XIetfEoSgK-knyrylAxwVo_QlhnxNrzs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("/content/Job_Placement_Data.csv")
df.head()

df

df.info()

d=df.iloc[:,[1, 3, 6, 9, 11,12]]
d.head()

plt.scatter(d['ssc_percentage'], d['hsc_percentage'],c=d['status'].astype('category').cat.codes)
plt.xlabel('SSC Percentage')
plt.ylabel('HSC Percentage')
plt.title('Scatter Plot: SSC vs HSC')
plt.show()

x=d.iloc[:,0:5]
y=d.iloc[:,-1]
x

y

from sklearn.model_selection import train_test_split as t

x_train,x_test,y_train,y_test=t(x, y, test_size=0.1)

x_train

y_train

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
x_train

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)

y_pred=lr.predict(x_test)

y_test

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

x_train.shape

import mlxtend
from mlxtend.plotting import plot_decision_regions
x_train_subset = x_train_df[['ssc_percentage', 'hsc_percentage']]
y_train_encoded = pd.Series(np.where(y_train.values == 'Placed', 1, 0), index=y_train.index)
lr = LogisticRegression()
lr.fit(x_train_subset_scaled, y_train_encoded)
plot_decision_regions(X=x_train_subset_scaled, y=y_train_encoded.values.ravel(), clf=lr, legend=2)

import pickle
pickle.dump(lr,open('model.pkl','wb'))