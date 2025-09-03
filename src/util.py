# The corrected function for util.py
import dill
import os
import sys
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            # Corrected line: Save to file_obj, not file_path
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train,y_train, X_test, y_test, models, params):
    try:
        report={}
        for i in range(len(list(models))):
            model=list(models.values())[i]
            para=params[list(models.keys())[i]]

            gs=GridSearchCV(model,para, cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)

            model_train_score=r2_score(y_train, y_train_pred)
            model_test_score=r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]]=model_test_score
        return report

    except Exception as e:
        raise CustomException(e,sys)