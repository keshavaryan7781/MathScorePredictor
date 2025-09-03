import os
import sys

from dataclasses import dataclass

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
    )
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from sklearn.neighbors import KNeighborsRegressor

from src.exception import CustomException
from src.util import save_object
from src.util import evaluate_model
from src.logger import logging

@dataclass
class ModelTrainingConfig:
    train_model_file_path= os.path.join('artifact','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainingConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting of training and testing set of data")
            
            X_train,y_train,X_test,y_test= (train_array[:,:-1],
                                            train_array[:,-1],
                                            test_array[:,:-1],
                                            test_array[:,-1])
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {

                    "Random Forest": {
                        "n_estimators": [100, 200],
                        "max_depth": [None, 10],
                    },

                    "Decision Tree": {
                        "max_depth": [None, 10],
                        "min_samples_split": [2, 5]
                    },

                    "Gradient Boosting": {
                        "n_estimators": [100, 200],
                        "learning_rate": [0.05, 0.1],
                        "max_depth": [3, 5]
                    },

                    "Linear Regression": {
                        "fit_intercept": [True, False]
                    },

                    "XGBRegressor": {
                        "n_estimators": [100, 200],
                        "learning_rate": [0.05, 0.1],
                        "max_depth": [3, 5]
                    },

                    "CatBoosting Regressor": {
                        "iterations": [100, 200],
                        "learning_rate": [0.05, 0.1],
                        "depth": [4, 6]
                    },

                    "AdaBoost Regressor": {
                        "n_estimators": [50, 100],
                        "learning_rate": [0.05, 0.1]
                    }
            }
            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)

            #get the best score of the trained models
            best_model_score=max(model_report.values())

            #name of the best model
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException(f"best model not found!!!")
            
            logging.info(f"The best model onn both the training and test datas is: {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.train_model_file_path,

                obj=best_model
            )

            pred=best_model.predict(X_test)

            r2_square=r2_score(y_test,pred)

            return r2_square
        

        except Exception as e:
            raise CustomException(e,sys)