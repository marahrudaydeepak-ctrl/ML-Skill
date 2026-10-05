from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report
d=fetch_openml('heart-statlog',version=1,as_frame=True); df=d.frame; target=d.target.name; y=df.pop(target).astype(str); X=df
num=X.select_dtypes(include='number').columns; cat=X.select_dtypes(exclude='number').columns
pre=ColumnTransformer([('num',SimpleImputer(strategy='median'),num),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),cat)])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y); p=Pipeline([('pre',pre),('tree',DecisionTreeClassifier(max_depth=5,min_samples_leaf=4,random_state=42))]); p.fit(Xtr,ytr); pred=p.predict(Xte)
print('Accuracy',accuracy_score(yte,pred)); print(classification_report(yte,pred))
