"""
------------- Population Research Centre RUG (PRC-RUG) ------------------------
Predicting Fertility data challenge (PreFer)
    Random forest model with hyper-parameter tuning and adaptive synthetic 
    sampling, theory-driven feature selection
-------------------------------------------------------------------------------
Version: June 1, 2024
         Updated based on the feedback of the PreFer team
         New model estimated based on a downgraded version of sklearn (due to unpickle issues)
         May 11: includes Abi's code and Zuzana's and Adrien's variables
         June 1: fixed script based on feedback to the previous code
         June 2: addtional interaction terms based on feature engineering
-------------------------------------------------------------------------------
"""  
from IPython import get_ipython
get_ipython().magic('reset -sf')
import joblib  
import pandas as pd
import sklearn
print(sklearn.__version__)
from sklearn.ensemble import RandomForestClassifier # Random forest
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import ADASYN
import numpy as np
import random
# import featurewiz as FW
# from sklearn.model_selection import train_test_split 
import os
os.chdir('Z:\\GitHub\\fertility-prediction-challenge\\')
#%% Loading data:
source = 'C:\\2024\\_April_2024\\Prefer\\02_consolidated\\'
consolidated_df = pd.read_csv(source + 'prefer_consolidated_db.csv', low_memory=False)
#%% Cleaning and data preparation function:
def clean_df(consolidated_df, background_df=None):  
    # Data preparation:
    # -------------------------------------------------------------------------
    # Duration and squared duration (years living with partner):
    consolidated_df['duration'] = 2020 - consolidated_df['cf20m029']
    consolidated_df['duration'].fillna(0, inplace=True)
    consolidated_df['duration2'] = consolidated_df['duration']*consolidated_df['duration']
        # Note: I am assuming missing duration is equal to 0 (not living with partner)
    # -------------------------------------------------------------------------
    # Recoding partner:
    consolidated_df['cf20m024'].value_counts()
    consolidated_df['cf20m025'].value_counts()
    consolidated_df['partner'] = 0
    consolidated_df.loc[(consolidated_df['cf20m024'] == 1) | (consolidated_df['cf20m025'] == 1), 'partner'] = 1
    consolidated_df['partner'].value_counts()
    # -------------------------------------------------------------------------
    # Satisfaction with partner:
    consolidated_df['partner_satisfaction'] = consolidated_df['cf20m180']
    consolidated_df['partner_satisfaction'].isna().sum()
    consolidated_df['partner_satisfaction'].fillna(0, inplace=True)
        # Note: we are assuming that people with not partners are unsatisfied 
    # -------------------------------------------------------------------------
    # Previous children
    consolidated_df['previous_children'] = consolidated_df['cf20m455'] #changed from 454 to 455
    consolidated_df['previous_children'].isna().sum()
    consolidated_df['previous_children'].fillna(0, inplace=True)
    consolidated_df['previous_children'].value_counts()
    # -------------------------------------------------------------------------    
    # Years since last child-Zuzana's code in python, recoded by Abi and ChatGPT LOL
    # Year of birth last child
    # Fifth child
    consolidated_df['last_child'] = np.where(consolidated_df['cf20m455'] == 5, consolidated_df['cf20m460'], np.nan)
    consolidated_df['last_child'] = np.where((consolidated_df['last_child'].isna()) & (consolidated_df['cf20m455'] == 5), consolidated_df['cf19l460'], consolidated_df['last_child'])
    consolidated_df['last_child'] = np.where((consolidated_df['last_child'].isna()) & (consolidated_df['cf20m455'] == 5), consolidated_df['cf18k460'], consolidated_df['last_child'])
    #Fourth
    consolidated_df['last_child'] = np.where(consolidated_df['cf20m455'] == 4, consolidated_df['cf20m459'], np.nan)
    consolidated_df['last_child'] = np.where((consolidated_df['last_child'].isna()) & (consolidated_df['cf20m455'] == 4), consolidated_df['cf19l459'], consolidated_df['last_child'])
    consolidated_df['last_child'] = np.where((consolidated_df['last_child'].isna()) & (consolidated_df['cf20m455'] == 4), consolidated_df['cf18k459'], consolidated_df['last_child'])
    #Third
    consolidated_df['last_child'] = np.where(consolidated_df['cf20m455'] == 3, consolidated_df['cf20m458'], np.nan)
    consolidated_df['last_child'] = np.where((consolidated_df['last_child'].isna()) & (consolidated_df['cf20m455'] == 3), consolidated_df['cf19l458'], consolidated_df['last_child'])
    consolidated_df['last_child'] = np.where((consolidated_df['last_child'].isna()) & (consolidated_df['cf20m455'] == 3), consolidated_df['cf18k458'], consolidated_df['last_child'])
    #Second
    consolidated_df['last_child'] = np.where(consolidated_df['cf20m455'] == 2, consolidated_df['cf20m457'], np.nan)
    consolidated_df['last_child'] = np.where((consolidated_df['last_child'].isna()) & (consolidated_df['cf20m455'] == 2), consolidated_df['cf19l457'], consolidated_df['last_child'])
    consolidated_df['last_child'] = np.where((consolidated_df['last_child'].isna()) & (consolidated_df['cf20m455'] == 2), consolidated_df['cf18k457'], consolidated_df['last_child'])
    #First
    consolidated_df['last_child'] = np.where(consolidated_df['cf20m455'] == 1, consolidated_df['cf20m456'], np.nan)
    consolidated_df['last_child'] = np.where((consolidated_df['last_child'].isna()) & (consolidated_df['cf20m455'] == 1), consolidated_df['cf19l456'], consolidated_df['last_child'])
    consolidated_df['last_child'] = np.where((consolidated_df['last_child'].isna()) & (consolidated_df['cf20m455'] == 1), consolidated_df['cf18k456'], consolidated_df['last_child'])
    # Number of years since last birth
    consolidated_df['zlast_child'] = 2020 - consolidated_df['last_child']
    consolidated_df['zlast_child'].isna().sum()
    consolidated_df['zlast_child'].value_counts()
    consolidated_df['zlast_child'].fillna(0, inplace=True)
    # -------------------------------------------------------------------------
    # Average age difference between father's and mother's age
    # Year of Mother
    # year of birth of mother 
    consolidated_df['cf20m009'].value_counts()
    consolidated_df['cf20m009'].isna().sum()
    consolidated_df['yrbirth_mom'] = consolidated_df['cf20m009']
    # Year of mother unknown and year of father known
    condition1 = consolidated_df['cf20m009'].isna() & consolidated_df['cf20m005'].notna()
    consolidated_df.loc[condition1, 'yrbirth_mom'] = consolidated_df.loc[condition1, 'cf20m005'] + 2.6
    # Impute mother's age with age of respondent for the rest
    mean_diff_mom = (consolidated_df['birthyear_bg'] - consolidated_df['cf20m009']).mean(skipna=True)
    consolidated_df['yrbirth_mom'] = consolidated_df['yrbirth_mom'].fillna(consolidated_df['birthyear_bg'] - mean_diff_mom - 0.9)
    # Variable of age of mother at birth of ego
    consolidated_df['motherage'] = consolidated_df['birthyear_bg'] - consolidated_df['yrbirth_mom']
    # Year of Father
    # Year of birth of Father
    consolidated_df['cf20m005'].value_counts()
    consolidated_df['cf20m005'].isna().sum()
    # Year of father unknown and year of mother known
    condition2 = consolidated_df['cf20m005'].isna() & consolidated_df['cf20m009'].notna()
    consolidated_df.loc[condition2, 'yrbirth_dad'] = consolidated_df.loc[condition2, 'cf20m009'] - 2.6
    # Impute dad's age with age of respondent for the rest
    mean_diff_dad = (consolidated_df['birthyear_bg'] - consolidated_df['cf20m005']).mean(skipna=True)
    consolidated_df['yrbirth_dad'] = consolidated_df['yrbirth_dad'].fillna(consolidated_df['birthyear_bg'] - mean_diff_dad)
    # Variable of age of father at birth of ego
    consolidated_df['fatherage'] = consolidated_df['birthyear_bg'] - consolidated_df['yrbirth_dad']
    # -------------------------------------------------------------------------
    # Financial satisfaction
    consolidated_df = consolidated_df.copy() # To reduce fragmentation
    consolidated_df['financial_satisfaction'] = consolidated_df['ci20m006']
    consolidated_df['financial_satisfaction'].isna().sum()
    consolidated_df['financial_satisfaction'].value_counts()
    consolidated_df['financial_satisfaction'].fillna(7, inplace=True)
        # Note: missing are replace by the mode
    # -------------------------------------------------------------------------
    # Attitudes
    # Respondents of Politics and Values Survey (1 is that they responded)
    consolidated_df['cvresp'] = 1
    consolidated_df.loc[consolidated_df['cv20l_m1'].isna() & consolidated_df['cv20l_m2'].isna() & consolidated_df['cv20l_m3'].isna(), 'cvresp']=0
    consolidated_df['cvresp'].value_counts(dropna=False)  
    #--------------------------------------------------------------------------
    # People that want to have children should get married.
    # 1 fully disagree; 2 disagree; 3 neither agree nor disagree; 4 agree; 5 fully agree
    #Recode to disagree, neither and agree (send unknowns to 3)
    consolidated_df['cv20l125'].value_counts(dropna=False)
    consolidated_df['getmarried_rec'] = consolidated_df['cv20l125']
    # 2 recoded to 1
    consolidated_df.loc[consolidated_df['cv20l125'] == 2, 'getmarried_rec'] = 1
    # 3 recoded to 2 
    consolidated_df.loc[consolidated_df['cv20l125'] == 3, 'getmarried_rec'] = 2
    #non reponse recoded to 2
    consolidated_df.loc[consolidated_df['cv20l125'].isna() & consolidated_df['cvresp'] == 1, 'getmarried_rec'] = 2
    #Recode 4 and 5 to 3
    consolidated_df.loc[consolidated_df['cv20l125'] == 4, 'getmarried_rec'] = 3
    consolidated_df.loc[consolidated_df['cv20l125'] == 5, 'getmarried_rec'] = 3
    #See values
    consolidated_df['getmarried_rec'].value_counts(dropna=False)      
    #--------------------------------------------------------------------------
    # "A single parent can raise a child just as well as two parents together."
    # 1 fully disagree; 2 disagree; 3 neither agree nor disagree; 4 agree; 5 fully agree
    #Recode to disagree, neither and agree (send unknowns to 3)
    consolidated_df['cv20l126'].value_counts(dropna=False)
    consolidated_df['singleparent_rec'] = consolidated_df['cv20l126']
    # 2 recoded to 1
    consolidated_df.loc[consolidated_df['cv20l126'] == 2, 'singleparent_rec'] = 1
    # 3 recoded to 2 
    consolidated_df.loc[consolidated_df['cv20l126'] == 3, 'singleparent_rec'] = 2
    #non reponse recoded to 2
    consolidated_df.loc[consolidated_df['cv20l126'].isna() & consolidated_df['cvresp'] == 1, 'singleparent_rec'] = 2
    #Recode 4 and 5 to 3
    consolidated_df.loc[consolidated_df['cv20l126'] == 4, 'singleparent_rec'] = 3
    consolidated_df.loc[consolidated_df['cv20l126'] == 5, 'singleparent_rec'] = 3
    #See values
    consolidated_df['singleparent_rec'].value_counts(dropna=False)   
    #--------------------------------------------------------------------------
    # "Both father and mother should contribute to the family income."
    # 1 fully disagree; 2 disagree; 3 neither agree nor disagree; 4 agree; 5 fully agree
    #Recode to disagree, neither and agree (send unknowns to 3)
    consolidated_df['cv20l112'].value_counts(dropna=False)
    consolidated_df['parcontribute_rec'] = consolidated_df['cv20l112']
    # 2 recoded to 1
    consolidated_df.loc[consolidated_df['cv20l112'] == 2, 'parcontribute_rec'] = 1
    # 3 recoded to 2 
    consolidated_df.loc[consolidated_df['cv20l112'] == 3, 'parcontribute_rec'] = 2
    #non reponse recoded to 2
    consolidated_df.loc[consolidated_df['cv20l112'].isna() & consolidated_df['cvresp'] == 1, 'parcontribute_rec'] = 2
    #Recode 4 and 5 to 3
    consolidated_df.loc[consolidated_df['cv20l112'] == 4, 'parcontribute_rec'] = 3
    consolidated_df.loc[consolidated_df['cv20l112'] == 5, 'parcontribute_rec'] = 3
    #See values
    consolidated_df['parcontribute_rec'].value_counts(dropna=False)
    #--------------------------------------------------------------------------
    # Health
    consolidated_df['ch20m004'].value_counts(dropna=False)
    consolidated_df['health'] = 0
    consolidated_df.loc[(consolidated_df['ch20m004'] == 3) | (consolidated_df['ch20m004'] == 4), 'health'] = 1
    consolidated_df['health'].value_counts(dropna=False)
    # -------------------------------------------------------------------------
    # One-hot encoding:
    consolidated_df = pd.get_dummies(consolidated_df, columns=['belbezig_2020'], prefix='employment')    
    consolidated_df = pd.get_dummies(consolidated_df, columns=['oplcat_2020'], prefix='eduencoded')
    consolidated_df = pd.get_dummies(consolidated_df, columns=['woning_2020'], prefix='dwelling')
    consolidated_df = pd.get_dummies(consolidated_df, columns=['cr20m142'], prefix='religious')
    consolidated_df = pd.get_dummies(consolidated_df, columns=['cr20m143'], prefix='church')
    consolidated_df = pd.get_dummies(consolidated_df, columns=['migration_background_bg'], prefix='migration')
    consolidated_df = pd.get_dummies(consolidated_df, columns=['burgstat_2020'], prefix='civilstatus')
    consolidated_df = pd.get_dummies(consolidated_df, columns=['cf20m128'], prefix='morechildren')
    consolidated_df = pd.get_dummies(consolidated_df, columns=['getmarried_rec'], prefix='getmarried')
    consolidated_df = pd.get_dummies(consolidated_df, columns=['singleparent_rec'], prefix='singleparent')
    consolidated_df = pd.get_dummies(consolidated_df, columns=['parcontribute_rec'], prefix='parcontribute')
    # -------------------------------------------------------------------------
    # Partner recoded after one-hot enconding:
    consolidated_df.rename(columns={'civilstatus_1.0': 'civilstatus_married'}, inplace=True)
    # Create a new DataFrame for the merged column
    # new_col = pd.DataFrame({
    #     'civilstatus_sepdivwid': (consolidated_df['civilstatus_2.0'] | consolidated_df['civilstatus_3.0'] | consolidated_df['civilstatus_4.0']).astype(int)
    # })
    new_col = pd.DataFrame({
        'civilstatus_sepdivwid': (consolidated_df['civilstatus_2.0'] | consolidated_df['civilstatus_3.0']).astype(int)
    })
        # Note: I excluded civil status 4 because it crashes the GitHub test
    consolidated_df = pd.concat([consolidated_df, new_col], axis=1)
    # Employment:
    consolidated_df.rename(columns={'employment_1.0': 'paid_employment'}, inplace=True)
    consolidated_df.rename(columns={'employment_7.0': 'studying_notemployed'}, inplace=True)
    # -------------------------------------------------------------------------    
    # Impute missing values of income with the median of the column
    new_columns = {
    'brutohh_f_2020_imputed_median': consolidated_df['brutohh_f_2020'].fillna(consolidated_df['brutohh_f_2020'].median()),
    'nettohh_f_2020_imputed_median': consolidated_df['nettohh_f_2020'].fillna(consolidated_df['nettohh_f_2020'].median())
    }
    consolidated_df = pd.concat([consolidated_df, pd.DataFrame(new_columns)], axis=1)
    # -------------------------------------------------------------------------
    # Feature engineering (SULOV algorithm and XGBoost):
    '''
    # Target and outcome
    y = cleaned_df['new_child'] # Target
    X = cleaned_df.drop(columns=['new_child','nomem_encr'])
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=666)
    # Merge X_train and y_train into a single dataframe
    train_data = X_train.copy()
    train_data['new_child'] = y_train.values
    # Merge X_test and y_test into a single dataframe
    test_data = X_test.copy()
    test_data['new_child'] = y_test.values
    outputs = FW.featurewiz(dataname=train_data, target='new_child', corr_limit=0.1, verbose=0, sep=',', 
    		header=0, test_data=test_data,feature_engg='interactions', category_encoders='',
    		dask_xgboost_flag=True, nrows=None)
    # Extract the column names 
    outputs[1].columns
    '''
    # -------------------------------------------------------------------------
    consolidated_df['motherage_mult_by_cvresp'] = consolidated_df['motherage']*consolidated_df['cvresp']
    # consolidated_df['motherage_mult_by_cvresp'].isna().sum()
    # consolidated_df['motherage_mult_by_cvresp'].value_counts()
    consolidated_df['financial_satisfaction_mult_by_motherage'] = consolidated_df['financial_satisfaction']*consolidated_df['motherage']
    # consolidated_df['financial_satisfaction_mult_by_motherage'].isna().sum()
    # consolidated_df['financial_satisfaction_mult_by_motherage'].value_counts(dropna=False)
    consolidated_df['zlast_child_mult_by_previous_children'] = consolidated_df['zlast_child']*consolidated_df['previous_children']
    # consolidated_df['zlast_child_mult_by_previous_children'].isna().sum()
    # consolidated_df['zlast_child_mult_by_previous_children'].value_counts(dropna=False)
    consolidated_df['fatherage_mult_by_cvresp'] = consolidated_df['fatherage']*consolidated_df['cvresp']
    consolidated_df['zlast_child_mult_by_partner'] = consolidated_df['zlast_child']*consolidated_df['partner']
    consolidated_df['health_mult_by_cvresp'] = consolidated_df['health']*consolidated_df['cvresp']
    # -------------------------------------------------------------------------
    # Define the list of column names to select
    selected_features = ['nomem_encr', 'new_child',
       'birthyear_bg', 
       'age_bg', 
       'gender_bg',
       'brutohh_f_2020_imputed_median', 
       'nettohh_f_2020_imputed_median',
       'eduencoded_6.0', 
       'eduencoded_5.0', 
       'eduencoded_4.0', 
       'financial_satisfaction',
       'dwelling_1.0', 
       'religious_5.0',
       'religious_6.0',
       'church_2.0', 
       'migration_0.0', 
       'health',
       'civilstatus_married',
       'previous_children', #Added
       'morechildren_1.0', 
       'morechildren_2.0', 
       'morechildren_3.0', 
       'getmarried_1.0',   # Recoded by Abi
       'getmarried_2.0',   # Recoded by Abi
       'singleparent_3.0', # Recoded by Abi
       'singleparent_2.0', # Recoded by Abi
       'parcontribute_3.0',# Recoded by Abi
       'parcontribute_2.0',# Recoded by Abi
       'paid_employment',
       'studying_notemployed',
       'duration', 
       'duration2',
       'partner_satisfaction',
       'motherage',     # Mother's age 
       'fatherage',     # Father's age 
       'zlast_child', # Number of years since last birth: Zuzana's variable in Python format (Abi)
       'partner', # with partner
       'cvresp', # Created by Abi
       'motherage_mult_by_cvresp', # interaction term obtained with feature engineering
       'financial_satisfaction_mult_by_motherage', # interaction term obtained with feature engineering
       'zlast_child_mult_by_previous_children', # interaction term obtained with feature engineering
       'fatherage_mult_by_cvresp', # interaction term obtained with feature engineering
       'zlast_child_mult_by_partner', # interaction term obtained with feature engineering
       'health_mult_by_cvresp' # interaction term obtained with feature engineering
       ]
    #%% Create a new DataFrame with the selected columns
    df = consolidated_df[selected_features].copy()
    means = df.mean()
    # Replace NaNs with the mean of their respective columns in case of any missing
    df.isna().sum()
    df = df.fillna(means)
    df.isna().sum()
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
    # Target and outcome
    y = cleaned_df['new_child'] # Target
    X = cleaned_df.drop(columns=['new_child','nomem_encr'])
    # Oversampling with ADASYN
    adasyn = ADASYN()
    Xsam, ysam = adasyn.fit_resample(X, y)
    # Define the parameter grid for RandomForestClassifier
    param_grid = {
        'n_estimators': [10, 20, 30, 40, 50, 75, 100],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 3, 5, 7, 10, 13],
        'min_samples_leaf': [1, 2, 3, 4, 6]
    }
    clf = RandomForestClassifier(n_jobs = -1)
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=cross_validations, scoring='f1')
    grid_search.fit(Xsam, ysam)
    best_clf = grid_search.best_estimator_
    predictions = best_clf.predict(X)
    joblib.dump(best_clf, 'RFmodelFE_June2.joblib')
    results = pd.DataFrame({
         'prediction': predictions,
         'new_child': y
     }) 
    return results

#%% ---------------------------------------------------------------------------
results = train_save_model(cleaned_df)
print(results) # Check that predictions are binary <-----------------------

#%% ---------------------------------------------------------------------------
# Scores
def score(results):
    # Merge predictions and ground truth on the 'id' column
    # merged_df = pd.merge(predictions_df, ground_truth_df, on="nomem_encr", how="right")
    merged_df = results
    # Calculate accuracy
    accuracy = len(merged_df[merged_df["prediction"] == merged_df["new_child"]]) / len(
        merged_df
    )

    # Calculate true positives, false positives, and false negatives
    true_positives = len(
        merged_df[(merged_df["prediction"] == 1) & (merged_df["new_child"] == 1)]
    )
    false_positives = len(
        merged_df[(merged_df["prediction"] == 1) & (merged_df["new_child"] == 0)]
    )
    false_negatives = len(
        merged_df[(merged_df["prediction"] == 0) & (merged_df["new_child"] == 1)]
    )

    # Calculate precision, recall, and F1 score
    try:
        precision = true_positives / (true_positives + false_positives)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = true_positives / (true_positives + false_negatives)
    except ZeroDivisionError:
        recall = 0
    try:
        f1_score = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1_score = 0
    # Write metric output to a new CSV file
    metrics_df = pd.DataFrame(
        {
            "accuracy": [accuracy],
            "precision": [precision],
            "recall": [recall],
            "f1_score": [f1_score],
        }
    )
    return metrics_df
#%% ---------------------------------------------------------------------------
performance_metrics = score(results)
print(performance_metrics) 