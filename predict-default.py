'''
This script runs a Flask web service to generate the 'default' prediction.

The script contains the following functions:
- data loading
- data preparation
- model training
- prediction
- save the prediction to a csv

Author: Kai Chen
Date: April 2019
'''


from flask import Flask, request, render_template, session, redirect

import io
import csv
import numpy as np
import pickle
import math

import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import seaborn as sns

from scipy import interp
from itertools import cycle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Normalizer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.externals import joblib

import statsmodels.formula.api as smf

import xgboost
from xgboost import XGBClassifier

from hyperopt import hp, tpe, STATUS_OK, Trials
from hyperopt.fmin import fmin

np.random.seed(42)


app = Flask(__name__, template_folder='templates')


data_path = "dataset.csv"

model_path = 'xgb.pkl'


def get_df(data_path):
    '''
    get dataframe from csv
    '''
    i = 0
    columns = []
    dict_data = {}
    with open(data_path, "r") as ins:
        for line in ins:
            if i == 0:
                columns = line.split(';')
                for col in columns:
                    dict_data[col] = []
            else:
                array = line.split(';')
                for j, el in enumerate(array):
                    dict_data[columns[j]].append(el)
            i += 1

    df = pd.DataFrame(dict_data)

    return df


def prepare_data(df):
    '''
    prepare dataset for machine learning models
    '''
    df.rename(columns={'"worst_status_active_inv"\n': '"worst_status_active_inv"'}, inplace=True)

    cat_columns = ['"account_status"', '"account_worst_status_0_3m"', '"account_worst_status_12_24m"',
                   '"account_worst_status_3_6m"', '"account_worst_status_6_12m"',
                   '"merchant_category"', '"merchant_group"',
                   '"name_in_email"',
                   '"status_last_archived_0_24m"', '"status_2nd_last_archived_0_24m"',
                   '"status_3rd_last_archived_0_24m"',
                   '"status_max_archived_0_6_months"', '"status_max_archived_0_12_months"',
                   '"status_max_archived_0_24_months"',
                   '"worst_status_active_inv"', '"has_paid"', '"default"']

    feature_columns = [col for col in df.columns.tolist() if col not in ['"uuid"', '"default"']]
    num_columns = [col for col in feature_columns if col not in cat_columns]

    df = df.replace('NA', np.nan, regex=True)

    for col in num_columns:
        df[col] = pd.to_numeric(df[col], errors='raise')

    for col in cat_columns:
        df[col] = df[col].astype('category').cat.as_ordered()

    # handle the missing values:
    # - for categorial variables, we don't need to do anything,
    # because pandas automatically convert NA to -1 for categorical variables.
    # - for continuous variables, we need to replace NA with mean or median.
    # - create a col_NA column to indicate which row has NAs.
    for col in df.columns:
        if is_numeric_dtype(df[col]):
            col_vals = df[col]
            if sum(col_vals.isnull()) != 0:
                df[col + '_na'] = col_vals.isnull()
                df[col] = col_vals.fillna(col_vals.median())

    # remove the highly correlated columns
    df_uncor = df.drop(['"max_paid_inv_0_24m"', '"num_arch_ok_12_24m"',
                        '"account_days_in_rem_12_24m"_na', '"account_days_in_term_12_24m"_na',
                        '"num_arch_written_off_0_12m"_na', '"num_arch_written_off_12_24m"_na'], axis=1)

    for col in cat_columns:
        if col != '"default"':
            label_encoder = LabelEncoder()
            label_encoder = label_encoder.fit(df_uncor[col].tolist())
            df_uncor[col] = label_encoder.transform(df_uncor[col].tolist())

    df_uncor_train = df_uncor[df_uncor['"default"'].notnull()]

    df_uncor_test = df_uncor[df_uncor['"default"'].isnull()]

    X_train = df_uncor_train.drop(['"default"', '"uuid"'], axis=1)
    y_train = df_uncor_train['"default"'].astype('category')

    X_test = df_uncor_test.drop(['"default"', '"uuid"'], axis=1)

    return X_train, y_train, X_test, df_uncor_train, df_uncor_test


def predict_xgb(X_train, y_train, X_test, df_pred,
                params={
                    'n_estimators': [5, 50, 100, 200],
                    # 'min_child_weight': [1, 5, 10],
                    # 'gamma': [0.5, 1, 1.5, 2, 5],
                    # 'subsample': [0.6, 0.8, 1.0],
                    # 'subsample': [0.6, 0.8, 1],
                    # 'colsample_bytree': [0.6, 0.8, 1.0],
                    # 'colsample_bytree': [0.6, 0.8, 1],
                    # 'max_depth': [3, 4, 5]
                    'max_depth': [2, 4, 6], },
                folds=3, param_comb=2, n_jobs=1):

    '''
    train a xgboost model on the train dataset and used the trained model to predict 'default'

    hyperparameter tuning
    https://machinelearningmastery.com/xgboost-python-mini-course/
    https://www.kaggle.com/tilii7/hyperparameter-grid-search-with-xgboost/code
    https://xgboost-clone.readthedocs.io/en/latest/parameter.html
    '''

    def timer(start_time=None):
        # fork from https://www.kaggle.com/tilii7/hyperparameter-grid-search-with-xgboost
        if not start_time:
            start_time = datetime.now()
            return start_time
        elif start_time:
            thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
            tmin, tsec = divmod(temp_sec, 60)
            print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

    xgb_clf = XGBClassifier(silent=True)

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    search = RandomizedSearchCV(xgb_clf,
                                param_distributions=params,
                                n_iter=param_comb,
                                scoring='roc_auc',  # binary classification
                                # scoring='accuracy',
                                # n_jobs=-1,
                                n_jobs=n_jobs,
                                cv=skf.split(X_train, y_train),
                                verbose=3, random_state=42)

    # search = GridSearchCV(model, params, scoring="neg_log_loss", n_jobs=-1, cv=skf)

    start_time = timer(None)
    search.fit(X_train, np.array(y_train.apply(int)))
    timer(start_time)

    print('--------------')
    print('\n all results:')
    print(search.cv_results_)

    print('\n best estimator:')
    print(search.best_estimator_)

    print('\n best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
    print(search.best_score_ * 2 - 1)

    print('\n best xgb hyperparameters:')
    print(search.best_params_)

    result_csv_path = 'xgb-search-results.csv'
    results = pd.DataFrame(search.cv_results_)
    results.to_csv(result_csv_path, index=False)
    print('save xgb search results to {}'.format(result_csv_path))

    model_path = 'xgb.pkl'
    joblib.dump(search.best_estimator_, model_path)
    # pickle.dump(search, open(model_path, "wb"))
    print('save xgb model to {}'.format(model_path))
    print('--------------')

    # search = joblib.load(model_path)
    y_pred = search.predict_proba(X_test)[:, 1]

    df_pred['"pd"'] = y_pred

    df_pred.to_csv('prediction.csv', index=False)

    print('save prediction to {}'.format('prediction.csv'))
    print('--------------')

    return df_pred

#@app.route("/")
@app.route("/prediction")
def get_prediction():
    '''
    This function generate the probability of 'default' and output the result to the html
    '''

    df = get_df(data_path)

    # get test data
    print('get test data')
    _, _, X_test, _, df_uncor_test = prepare_data(df)

    # load the trained model
    print('load model')
    search = joblib.load(model_path)

    # make prediction
    print('make prediction')
    y_pred = search.predict_proba(X_test)[:, 1]

    # save the prediction
    df_pred = df_uncor_test[['"uuid"']]
    df_pred['"pd"'] = y_pred
    df_pred.to_csv('prediction.csv', index=False)

    print('save prediction to {}'.format('prediction.csv'))
    print('done')

    #df_pred = predict_xgb(X_train, y_train, X_test, df_pred)


    #https://galaxydatatech.com/2018/03/31/passing-dataframe-web-page/
    #https: // stackoverflow.com / questions / 52644035 / how - to - show - a - pandas - dataframe - into - a - existing - flask - html - table
    #http: // flask.pocoo.org / docs / 1.0 / quickstart /  # rendering-templates
    # https://stackoverflow.com/questions/23327293/flask-raises-templatenotfound-error-even-though-template-file-exists/23327352
    return render_template('prediction.html', tables=[df_pred.to_html(classes='data')], titles=df.columns.values)

    #return render_template("prediction.html", data=df_pred.to_html())
    #return render_template('prediction.html', tables=[df_pred.to_html(classes='data')], titles=df_pred.columns.values)


if __name__ == '__main__':
    app.run(debug=True)