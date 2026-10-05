import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
df=sns.load_dataset('titanic')
print(df.head()); print(df.isna().sum()); print(df.describe(include='all').T)
X=df.drop(columns=['survived','alive','class','deck'],errors='ignore'); y=df.survived
num=X.select_dtypes(include='number').columns; cat=X.select_dtypes(exclude='number').columns
pre=ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),cat)])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
model=Pipeline([('pre',pre),('clf',LogisticRegression(max_iter=1000))]); model.fit(Xtr,ytr); p=model.predict(Xte)
print('Accuracy:',accuracy_score(yte,p)); print(classification_report(yte,p))
