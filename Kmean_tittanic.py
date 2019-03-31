from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

#names=['preg','plas','pres','skin','test','mass','pedi','age','binary']
#dataframe=read_csv("pima-indians-diabetes.data.csv",names=names)
dataframe=read_csv("train_titanic.csv")
dataframe=dataframe.drop(columns=['Name','Sex','Age','Ticket','Cabin','Embarked'],axis=1)
array=dataframe.values
Y=array[:,1]
print(type(Y))
dataframe=dataframe.drop(columns='Survived',axis=1)
print(dataframe)
X=dataframe.values

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=8)
print(X_train)
kmeans=KMeans(n_clusters=3)
print("kmean",kmeans)
kmeans=kmeans.fit(X_train)
labels=kmeans.predict(X_train)
print(labels)
print(len(labels))
print(len(X_train))
print(type(labels))
centers=kmeans.cluster_centers_
print(centers)

labels_test=kmeans.predict(X_test)
print(labels_test)