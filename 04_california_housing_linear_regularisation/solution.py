import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
df=pd.read_csv('dataset.csv'); X=df.drop(columns='median_house_value'); y=df.median_house_value
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
for name,m in [('Linear',LinearRegression()),('Ridge',Ridge(alpha=10)),('Lasso',Lasso(alpha=100)),('ElasticNet',ElasticNet(alpha=.1,l1_ratio=.5))]:
 p=Pipeline([('scale',StandardScaler()),('model',m)]); p.fit(Xtr,ytr); q=p.predict(Xte); print(name,mean_squared_error(yte,q)**.5,mean_absolute_error(yte,q),r2_score(yte,q))
