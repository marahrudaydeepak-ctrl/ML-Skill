import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
df=pd.read_csv('dataset.csv');X=df.drop(columns='target');y=df.target;a,b,c,d=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
models={'XGBoost':XGBClassifier(n_estimators=100,max_depth=3,learning_rate=.05,eval_metric='logloss',random_state=42),'LightGBM':LGBMClassifier(n_estimators=100,max_depth=4,learning_rate=.05,verbosity=-1,random_state=42)}
for n,m in models.items():m.fit(a,c);print(n,accuracy_score(d,m.predict(b)))
try:
 import shap,matplotlib.pyplot as plt
 e=shap.TreeExplainer(models['XGBoost']);v=e.shap_values(b);shap.summary_plot(v,b,show=False);plt.tight_layout();plt.savefig('plots/shap_summary.svg');plt.close()
except Exception as ex:print('SHAP:',ex)
