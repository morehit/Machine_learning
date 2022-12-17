import sklearn 
import pandas as pd
import numpy as np 

from sklearn import datasets
from sklearn.feature_selection import VarianceThreshold
 
 
data = {
      "A" : [1, 2, 4, 1 , 2 ],
      "B" : [4 ,5 ,6 ,7 ,8 ], 
      "C" : [1 ,1 ,1 ,1 ,1 ], 
      "D" : [0 ,0 ,0 , 0 , 0 ]
}

df = pd.DataFrame(data) 

print(df)

thresholder = VarianceThreshold(threshold=0)

X_high_variance = thresholder.fit_transform(df)
print(X_high_variance)
