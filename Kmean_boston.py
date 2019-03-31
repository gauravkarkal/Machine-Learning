from sklearn.model_selection import train_test_split
import pandas as pd
from pandas import read_csv
from sklearn.cluster import KMeans

#filename='housing.csv'
names=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PIRATIO','B','LSTAT','MEDV']
dataframe=read_csv("housing_original.csv",delim_whitespace=True,names=names)

array=dataframe.values
array=dataframe.values
X=array[:,0:13]
Y=array[:,13]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=7)

kmeans=KMeans(n_clusters=3)
print("kmean",kmeans)
kmeans=kmeans.fit(X_train)
labels=kmeans.predict(X_train)
print(labels)
print(len(labels))
print(type(labels))
centers=kmeans.cluster_centers_
print(centers)












































































































































