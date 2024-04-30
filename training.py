"""
--------------- Population Research Centre (RUG) ------------------------------
Predicting Fertility data challenge (PreFer)
-------------------------------------------------------------------------------
"""
from IPython import get_ipython
get_ipython().magic('reset -sf')
import joblib  # Ensure this import is at the top of your script
import pandas as pd
from sklearn.ensemble import RandomForestClassifier # Random forest
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import ADASYN
import numpy as np
import random
import os
os.chdir('Z:\\GitHub\\fertility-prediction-challenge\\')
#%% Loading data:
source = 'C:\\2024\\_April_2024\\Prefer\\02_consolidated\\'
consolidated_df = pd.read_csv(source + 'prefer_consolidated_db.csv', low_memory=False)
#%% Cleaning function:
def clean_df(consolidated_df, background_df=None):
    # One-hot encoding:
    consolidated_df = pd.get_dummies(consolidated_df, columns=['oplcat_2020'], prefix='eduencoded')
    consolidated_df = pd.get_dummies(consolidated_df, columns=['cf20m024'], prefix='partner')
    consolidated_df = pd.get_dummies(consolidated_df, columns=['cf20m454'], prefix='prevchild')
    consolidated_df = pd.get_dummies(consolidated_df, columns=['ci20m006'], prefix='finsatisf')
    consolidated_df = pd.get_dummies(consolidated_df, columns=['woning_2020'], prefix='dwelling')
    consolidated_df = pd.get_dummies(consolidated_df, columns=['cr20m142'], prefix='religious15')
    consolidated_df = pd.get_dummies(consolidated_df, columns=['cr20m143'], prefix='church')
    consolidated_df = pd.get_dummies(consolidated_df, columns=['migration_background_bg'], prefix='migration')
    consolidated_df = pd.get_dummies(consolidated_df, columns=['ch20m004'], prefix='health')
    consolidated_df = pd.get_dummies(consolidated_df, columns=['cf20m029'], prefix='livwpartner')
    consolidated_df = pd.get_dummies(consolidated_df, columns=['burgstat_2020'], prefix='civilstatus')
    consolidated_df = pd.get_dummies(consolidated_df, columns=['cf20m180'], prefix='satisfpartner')
    consolidated_df = pd.get_dummies(consolidated_df, columns=['cf20m128'], prefix='morechildren')
    consolidated_df = pd.get_dummies(consolidated_df, columns=['cv20l125'], prefix='getmarried')
    consolidated_df = pd.get_dummies(consolidated_df, columns=['cv20l126'], prefix='singleparent')
    consolidated_df = pd.get_dummies(consolidated_df, columns=['cv20l112'], prefix='parcontribute')
    # Impute missing values with the median of the column
    new_columns = {
    'brutohh_f_2020_imputed_median': consolidated_df['brutohh_f_2020'].fillna(consolidated_df['brutohh_f_2020'].median()),
    'nettohh_f_2020_imputed_median': consolidated_df['nettohh_f_2020'].fillna(consolidated_df['nettohh_f_2020'].median())
    }
    consolidated_df = pd.concat([consolidated_df, pd.DataFrame(new_columns)], axis=1)
    # Seleting relevant variables:
    # Define the list of column names to select
    selected_features = ['nomem_encr',
                         'new_child', # target
                        'birthyear_bg',  
                        'age_bg',
                         'gender_bg',
                        'brutohh_f_2020_imputed_median',
                        'nettohh_f_2020_imputed_median',
                        'eduencoded_1.0',  
                         'finsatisf_1.0',
                         'dwelling_1.0',
                         'religious15_1.0',
                         'church_1.0',
                         'migration_0.0',
                         'livwpartner_2015.0',
                         'civilstatus_1.0',
                         'satisfpartner_10.0',
                         'morechildren_1.0',
                         'getmarried_1.0',
                         'singleparent_1.0']
    # Create a new DataFrame with the selected columns
    df = consolidated_df[selected_features].copy()
    return df
#%% --------------------------------------------------------------------------
cleaned_df = clean_df(consolidated_df, background_df=None)
cleaned_df.head(10)
print(cleaned_df.columns)
#%% --------------------------------------------------------------------------
def train_save_model(cleaned_df):
    # Seed for reproducibility
    np.random.seed(666)  # Seed for NumPy operations
    random.seed(666)  # Seed for Python random module
    cross_validations = 10
    y = cleaned_df['new_child'] # Target
    X = cleaned_df.drop(columns=['new_child','nomem_encr'])
    # oversampling with ADASYN
    adasyn = ADASYN()
    Xsam, ysam = adasyn.fit_resample(X, y)
    # Define the parameter grid for RandomForestClassifier
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    clf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=cross_validations, scoring='f1')
    grid_search.fit(Xsam, ysam)
    best_clf = grid_search.best_estimator_
    best_clf.fit(Xsam, ysam)
    predictions = best_clf.predict(Xsam)
    joblib.dump(best_clf, 'random_forest_model.joblib')
    return predictions
#%% ------------------------------------------------------------------
predictions = train_save_model(cleaned_df)
print(predictions)