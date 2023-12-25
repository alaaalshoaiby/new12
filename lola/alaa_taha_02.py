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

plt.title('تطور الناتج المحلي الاجمالي على مدى السنوات')
plt.xlabel('السنه')
plt.ylabel('الناتج المحلي الاجمالي(بالمليارات من الدولار(')

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x,y)

print(model.coef_)
print(model.intercept_)

plt.scatter(x,y)
plt.plot(x,y,'r')

model.predict([[8]])

model.score(x, y)
