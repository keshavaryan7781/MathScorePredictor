import sys
import os

import numpy as np
import pandas as pd

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.util import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifact','preprocessor.pkl')
class DataTransformation:
    def __init__(self):
        self.data_transformation_config= DataTransformationConfig()
    def get_data_transformation_obj(self):

        try:
            numerical_features=[
                "writing_score","reading_score"
            ]
            categorical_features=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]


            logging.info(f"The numerical features are: {numerical_features}")
            logging.info(f"The categorical features are: {categorical_features}")  
            num_pipeline=Pipeline(
                [("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                [
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )  

            logging.info("numerical imputing and scaling is done")
            logging.info("categorical imputing, encoding and scaling is done")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_features),
                ("cat_ppeline",cat_pipeline,categorical_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        
        try:
            
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read the train & test data is completed")

            preprocessing_obj=self.get_data_transformation_obj()

            target_column_name="math_score"

            numerical_column=['writing_score','reading_score']
            input_feature_train_df=train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying preprocessor on test dataframe and training dataframe")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr=np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("saving the preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,

                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path

            )

        except Exception as e:
            raise CustomException(e,sys)
        