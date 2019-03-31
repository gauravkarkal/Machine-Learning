from sklearn.model_selection import train_test_split
import pandas as pd
from pandas import read_csv
from sklearn.cluster import KMeans
names=['preg','plas','pres','skin','test','mass','pedi','age','binary']
dataframe=read_csv("pima-indians-diabetes.data.csv",names=names)
array=dataframe.values
#pd.DataFrame(dataframe)
X=array[:,0:8]
Y=array[:,8]
alist=[1,2,3]
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