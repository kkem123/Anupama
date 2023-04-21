import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

classification_data=pd.read_excel("/Users/hzpro/acv/setosa/iris.xls")
classification_data.head()

x = classification_data.drop(['Classification'], axis=1)
y = classification_data['Classification']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.linear_model import LogisticRegression
reg=LogisticRegression()
reg=reg.fit(x_train,y_train)
preds=reg.predict(x_test)

model_pkl = open('model.pkl','wb')
#pickle.dump(cb_clf,CB_pkl)
pickle.dump(reg,model_pkl)
model_pkl.close()

#pickle.dump(reg,open('model.pkl','wb'))

