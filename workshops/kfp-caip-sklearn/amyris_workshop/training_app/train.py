
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

import hypertune

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

def train_evaluate(job_dir, training_dataset_path, alpha, max_iter, hptune):
    data = pd.read_excel(training_dataset_path,sheet_name='data')
    meta_data = pd.read_excel(training_dataset_path, sheet_name='meta data')
    
    numeric_vars = ((data.dtypes == 'float64') | (data.dtypes == 'int64')) & (meta_data['variable type'] == 'independent').values
    numeric_x_data = data[data.columns[numeric_vars]]

    #things to try to predict
    y_data = data[['Run_Performance']]
    meta_data = meta_data.set_index('name')
    y_data.Run_Performance.replace(('delta', 'gamma'), (1, 0), inplace=True)

    X_train = numeric_x_data[:1400]
    y_train = y_data[:1400]
    X_validation = numeric_x_data[1400:]
    y_validation = y_data[1400:]
    
    if not hptune:
        X_train = pd.concat([X_train, X_validation])
        y_train = pd.concat([y_train, y_validation])

    #impute missing with median
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')

    #auto scale
    scaler = StandardScaler()
    pca = PCA(n_components=3)
    pipe = Pipeline([('imputer',imputer),
                     ('scaler', scaler),
                     ('pca', pca),
                     ('classifier', SGDClassifier(loss='log', tol=1e-3))
                    ])

    

    print('Starting training: alpha={}, max_iter={}'.format(alpha, max_iter))

    pipe.set_params(classifier__alpha=alpha, classifier__max_iter=max_iter)
    pipe.fit(X_train, y_train.values.ravel())

    if hptune:
        accuracy = pipe.score(X_validation, y_validation)
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
