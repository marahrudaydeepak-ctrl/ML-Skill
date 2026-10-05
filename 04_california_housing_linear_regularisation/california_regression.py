import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
X,y=fetch_california_housing(return_X_y=True,as_frame=True); Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
models={'Linear':LinearRegression(),'Ridge':Ridge(1.0),'Lasso':Lasso(.001,max_iter=10000),'ElasticNet':ElasticNet(.001,.5,max_iter=10000)}
for name,m in models.items():
    p=Pipeline([('scale',StandardScaler()),('model',m)]); p.fit(Xtr,ytr); pred=p.predict(Xte); print(name,'MAE',mean_absolute_error(yte,pred),'RMSE',np.sqrt(mean_squared_error(yte,pred)),'R2',r2_score(yte,pred))
