# -*- coding: utf-8 -*-
"""
--------------- Population Research Centre (RUG) ------------------------------
Predicting Fertility data challenge (PreFer)
-------------------------------------------------------------------------------
"""
# List your libraries and modules here. Don't forget to update environment.yml!
from IPython import get_ipython
get_ipython().magic('reset -sf')
import pandas as pd
# ----------------------------------------------------------------------------
def clean_df(dfy,dfx, background_df=None):
    model_df = pd.merge(dfx,dfy, on="nomem_encr")
    # Filter cases for whom the outcome is not available
    consolidated_df = model_df[~model_df['new_child'].isna()]  
    #%% One-hot encoding:
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
    # Define the list of column names to select
    selected_features = ['nomem_encr', 'new_child', 'birthyear_bg', 'age_bg', 'gender_bg',
       'brutohh_f_2020_imputed_median', 'nettohh_f_2020_imputed_median',
       'eduencoded_1.0', 'finsatisf_1.0', 'dwelling_1.0', 'religious15_1.0',
       'church_1.0', 'migration_0.0', 'livwpartner_2015.0', 'civilstatus_1.0',
       'satisfpartner_10.0', 'morechildren_1.0', 'getmarried_1.0',
       'singleparent_1.0']
    #%% Create a new DataFrame with the selected columns
    df = consolidated_df[selected_features].copy()
    return df
# Sources:
source = 'Z:\\GitHub\\fertility-prediction-challenge\\'
dfy = pd.read_csv(source + 'PreFer_fake_outcome.csv')
dfx = pd.read_csv(source + 'PreFer_fake_data.csv')
df_clean = clean_df(dfx,dfy,background_df=None)


def predict_outcomes(dfx, dfy, background_df=None, model_path="Z:\\GitHub\\fertility-prediction-challenge\\random_forest_model.joblib"):
    df = clean_df(dfx, dfy, background_df=None)
    df = df.drop(columns='new_child')
    means = df.mean()
    # Replace NaNs with the mean of their respective columns
    df = df.fillna(means)
    ## This script contains a bare minimum working example
    if "nomem_encr" not in df.columns:
        print("The identifier variable 'nomem_encr' should be in the dataset")
    # Load the model
    import joblib
    model = joblib.load(model_path)
    # Exclude the variable nomem_encr if this variable is NOT in your model
    vars_without_id = df.columns[df.columns != 'nomem_encr']
    # Generate predictions from model, should be 0 (no child) or 1 (had child)
    predictions = model.predict(df[vars_without_id])
    # Output file should be DataFrame with two columns, nomem_encr and predictions
    df_predict = pd.DataFrame(
        {"nomem_encr": df["nomem_encr"], "prediction": predictions}
    )
    # Return only dataset with predictions and identifier
    return df_predict

dfx = pd.read_csv(source + 'PreFer_fake_data.csv')
dfy = pd.read_csv(source + 'PreFer_fake_outcome.csv')
yholdout = predict_outcomes(dfx, dfy, background_df=None, model_path="Z:\\GitHub\\fertility-prediction-challenge\\random_forest_model.joblib")
