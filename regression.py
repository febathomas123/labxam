import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
dataset=pd.read_csv('slr.csv')
a=dataset.iloc[:,:1].values
b=dataset.iloc[:,2].values
a_train,a_test,b_train,b_test=train_test_split(a,b,test_size=0.3,random_state=42)
model=LinearRegression()
model.fit(a_train,b_train)
c=model.predict(a_test)
print("predicted a_test:",c)
plt.scatter(a_train,b_train,color="blue")
plt.plot(a_train,model.predict(a_train),color="red")
plt.show()
from sklearn import metrics
ame=metrics.mean_absolute_error(b_test,c)
mse=metrics.mean_squared_error(b_test,c)
rmse=np.sqrt(mse)
