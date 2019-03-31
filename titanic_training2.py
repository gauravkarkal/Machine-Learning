from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#names=['preg','plas','pres','skin','test','mass','pedi','age','binary']
#dataframe=read_csv("pima-indians-diabetes.data.csv",names=names)
dataframe=read_csv("train_titanic.csv")
dataframe=dataframe.drop(columns=['Name','Sex','Ticket','Cabin','Embarked'],axis=1)
dataframe.dropna(inplace=True)
array=dataframe.values
Y=array[:,1]

###############################################################Y=Y.astype('int') to convert Y to int from object
dataframe=dataframe.drop(columns='Survived',axis=1)
print(dataframe)
X=dataframe.values

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=7)
print(X_train,X_test,Y_train,Y_test)

model=LogisticRegression()
model.fit(X_train,Y_train)
result=model.score(X_test,Y_test)
print("Accuracy",result*100.0)
