import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
df=sns.load_dataset('titanic'); X=df.drop(columns=['survived','alive','class','deck'],errors='ignore'); y=df.survived
num=X.select_dtypes(include='number').columns; cat=X.select_dtypes(exclude='number').columns
pre=ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),cat)])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y); p=Pipeline([('pre',pre),('model',LogisticRegression(max_iter=2000))]); p.fit(Xtr,ytr); prob=p.predict_proba(Xte)[:,1]
for t in [.3,.5,.7]:
    pred=(prob>=t).astype(int); print(t,accuracy_score(yte,pred),precision_score(yte,pred),recall_score(yte,pred),f1_score(yte,pred))
print('ROC-AUC',roc_auc_score(yte,prob)); print(classification_report(yte,(prob>=.5).astype(int)))
