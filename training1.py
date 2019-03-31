from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
names=['preg','plas','pres','skin','test','mass','pedi','age','binary']
dataframe=read_csv("pima-indians-diabetes.data.csv",names=names)
array=dataframe.values
X=array[:,0:8]
Y=array[:,8]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=7)
#print(X_train,X_test,Y_train,Y_test)



model=LogisticRegression()
model.fit(X_train,Y_train)
result=model.score(X_test,Y_test)
print("Accuracy",result*100.0)

predicted=model.predict(X_test)
print(predicted)
matrix=confusion_matrix(Y_test,predicted)
print(matrix)