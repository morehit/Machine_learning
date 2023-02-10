#Importing the libraries
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 

#Importing the Dataset
dataset = pd.read_csv('Data.csv') 
X = dataset.iloc[:, :-1].values 
Y = dataset.iloc[:, -1].values 

#Splitting the data into training and testing set 
from sklearn.model_selection import train_test_split 
X_train , X_test ,Y_train , Y_test = train_test_split(X,Y,test_size = 0.2 , random_state = 1 )

# Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() ; 
regressor.fit(X_train , Y_train)

#prediction the test results 
y_pred = regressor.predict(X_test)


# Visualising the Training set results
plt.scatter(X_train , Y_train , color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue' )
plt.title('Salary vs Experience(Training set)')
plt.xlabel('years of Experience ')
plt.ylabel('Salary')
plt.show()

# Visualising the Training set results
plt.scatter(X_test , Y_test , color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue' )
plt.title('Salary vs Experience(Training set)')
plt.xlabel('years of Experience ')
plt.ylabel('Salary')
plt.show()
