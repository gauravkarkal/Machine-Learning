import numpy as np
import pandas as py
data=py.read_csv("test_titanic.csv")

print(data.describe())
print(data.shape)

print(data.head(10))
print(data.tail(10))


#import matplotlib.pyplot as plt
#data.plot()
#plt.show()
