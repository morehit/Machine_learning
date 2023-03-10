#Importing the libraries
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 

#Importing the Dataset
dataset = pd.read_csv('Data.csv') 
X = dataset.iloc[:, :-1].values 
Y = dataset.iloc[:, -1].values 

#Taking care of missing data 
from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(missing_values = np.nan , strategy = 'mean')
imputer.fit(X[:, 1:3])
X[:,1:3] = imputer.transform(X[:,1:3])
print(X)

#Encoding independent variable 
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))
print(X)

#Encoding dependent variable 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)

print(Y)

#splitting the data into training and testing set 
from sklearn.model_selection import train_test_split
X_train, X_test , Y_train, Y_test = train_test_split(X,Y, test_size = 0.2 , random_state =1 )

#feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
