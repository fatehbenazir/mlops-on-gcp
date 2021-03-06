
# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
import sys

import fire
import pickle
import numpy as np
import pandas as pd
import json

import hypertune

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from googleapiclient import discovery
from googleapiclient import errors


def train_evaluate(job_dir, training_dataset_path, n_estimators, max_leaf_nodes, max_depth, min_samples_split, max_features, 
                   class_weight, bootstrap, hptune):
    data = pd.read_excel(training_dataset_path,sheet_name='data')
    meta_data = pd.read_excel(training_dataset_path, sheet_name='meta data')
    
    numeric_vars = ((data.dtypes == 'float64') | (data.dtypes == 'int64')) & (meta_data['variable type'] == 'independent').values
    numeric_x_data = data[data.columns[numeric_vars]]

    model_target = 'Run_Performance'
    y_data = data[[model_target]] 
    meta_data = meta_data.set_index('name')
 

    #maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(numeric_x_data, y_data, test_size=0.25, stratify = y_data[model_target], random_state=42)

    #split train set to create a pseudo test or validation dataset
    X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.33, stratify= y_train[model_target], random_state=42)
    
    if not hptune:
        X_train = pd.concat([X_train, X_validate])
        y_train = pd.concat([y_train, y_validate])

    #impute missing with median
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    #auto scale
    scaler = StandardScaler()
    
     # Simple hyperparameter tunnig for random forest model
    estimator = RandomForestClassifier(random_state = 42)
    
     
    pipe = Pipeline([('imputer',imputer),
                     ('scaler', scaler),
                     ('rfclassifier', estimator)
                    ])
    
    
    pipe.set_params(rfclassifier__n_estimators=n_estimators, rfclassifier__max_leaf_nodes=max_leaf_nodes, rfclassifier__max_depth=max_depth,
                       rfclassifier__min_samples_split=min_samples_split, rfclassifier__max_features=max_features, 
                       rfclassifier__class_weight=class_weight, rfclassifier__bootstrap=bootstrap )
    pipe.fit(X_train, y_train)
    
    if hptune:
        accuracy = pipe.score(X_validate, y_validate)
        print('Model accuracy: {}'.format(accuracy))
        # Log it with hypertune
        hpt = hypertune.HyperTune()
        hpt.report_hyperparameter_tuning_metric(
          hyperparameter_metric_tag='accuracy',
          metric_value=accuracy
        )

    # Save the model
    if not hptune:
        model_filename = 'model.pkl'
        with open(model_filename, 'wb') as model_file:
            pickle.dump(pipe, model_file)
        gcs_model_path = "{}/{}".format(job_dir, model_filename)
        subprocess.check_call(['gsutil', 'cp', model_filename, gcs_model_path], stderr=sys.stdout)
        print("Saved model in: {}".format(gcs_model_path)) 
    
if __name__ == "__main__":
    fire.Fire(train_evaluate)
