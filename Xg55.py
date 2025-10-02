import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,classification_report

#load dataset
data=load_breast_cancer()
X,y=data.data,data.target
#split data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#display dataset info
print(f"Features:{data.feature_names}")
print(f"Classes:{data.target_names}")

#convert dataset to DMatrix
dtrain=xgb.DMatrix(X_train,label=y_train)
dtest=xgb.DMatrix(X_test,label=y_test)

#train xgboost model
params={
    'objective':'binary:logistic',
    'eval_metric':'logloss',
    'max_depth':3,
    'eta':0.1
}
xgb_model=xgb.train(params,dtrain,num_boost_round=100)
#predict
y_pred=(xgb_model.predict(dtest)>0.5).astype(int)
#evaluate performance
accuracy=accuracy_score(y_test,y_pred)
print(f"XGBoost Accuracy:{accuracy}")
print("\n Classification Report: \n",classification_report(y_test,y_pred))