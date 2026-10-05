import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
X,y=load_diabetes(return_X_y=True,as_frame=True); q=y.quantile([1/3,2/3]).values; yc=np.where(y<=q[0],0,np.where(y<=q[1],1,2))
Xtr,Xte,ytr,yte=train_test_split(X,yc,test_size=.2,random_state=42,stratify=yc)
p=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(multi_class='multinomial',max_iter=2000))]); p.fit(Xtr,ytr); pred=p.predict(Xte)
print('0=Low 1=Medium 2=High'); print(classification_report(yte,pred)); print(confusion_matrix(yte,pred))
