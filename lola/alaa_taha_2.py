import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('economic_data.csv')
print(data)

plt.scatter(data['Year'], data['GDP'])
plt.show()


x=data.iloc[:,:1]
y=data.iloc[:,1]
 
print(x)
print(y)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x,y)

print(model.coef_)
print(model.intercept_)

plt.scatter(x,y)
plt.plot(x,model.predict(x),'r')

model.predict([[8]])

model.score(x, y)
