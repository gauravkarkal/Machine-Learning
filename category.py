import pandas as pd
import seaborn as sns
import numpy as np
data_train=pd.read_csv('train_titanic.csv')

print(data_train.sample(3))

print(data_train.describe())
#sns.barplot(x="Embarked",y="Survived",hue="Sex",data=data_train)

df=data_train
print(df.Age)
df.Age=df.Age.fillna(-0.5)
print(df.Age)

bins=(-1,0,5,12,18,25,35,60,120)
group_names=['Unknown','Baby','Child','Teenager','Student','Young adult','Adult','Senior']
categories=pd.cut(df.Age,bins,labels=group_names)
categories=np.array(categories)
df.Age=categories
print(df.Age)

