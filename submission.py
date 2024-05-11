# -*- coding: utf-8 -*-
         
"""
------------- Population Research Centre RUG (PRC-RUG) ------------------------
Predicting Fertility data challenge (PreFer)
    Random forest model with hyper-parameter tuning and adaptive synthetic 
    sampling, theory-driven feature selection
-------------------------------------------------------------------------------
Version: May 11, 2024
         Updated based on the feedback of the PreFer team
         New model estimated based on a downgraded version of sklearn (due to unpickle issues)
         May 11: includes Abi's code and Zuzana's and Adrien's variables
-------------------------------------------------------------------------------
"""         
# List your libraries and modules here. Don't forget to update environment.yml!
import pandas as pd
# ----------------------------------------------------------------------------
def clean_df(df, background_df=None):
    # Filter cases for whom the outcome is not available
    consolidated_df = df.loc[df['outcome_available'] == 1]  
    #%% Data preparation:
    # -------------------------------------------------------------------------
    # Duration and squared duration (years living with partner):
    consolidated_df['duration'] = 2020 - consolidated_df['cf20m029']
    consolidated_df['duration'].fillna(0, inplace=True)
    consolidated_df['duration2'] = consolidated_df['duration']*consolidated_df['duration']
        # Note: I am assuming missing duration is equal to 0 (not living with partner)
    # -------------------------------------------------------------------------
    # Recoding partner based on Zuzana's script:
    consolidated_df['partner_rec'] = consolidated_df['cf20m024']
    consolidated_df.loc[consolidated_df['cf20m025'] == 2, 'partner_rec'] = 3
    consolidated_df['partner_rec'].value_counts()
    # -------------------------------------------------------------------------
    # Satisfaction with partner:
    consolidated_df['partner_satisfaction'] = consolidated_df['cf20m180']
    consolidated_df['partner_satisfaction'].isna().sum()
    consolidated_df['partner_satisfaction'].fillna(0, inplace=True)
        # Note: we are assuming that people with not partners are unsatisfied 
    # -------------------------------------------------------------------------
    # Previous children
    consolidated_df['previous_children'] = consolidated_df['cf20m454']
    consolidated_df['previous_children'].isna().sum()
    consolidated_df['previous_children'].fillna(0, inplace=True)
    consolidated_df['previous_children'].value_counts()
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
    #People that want to have children should get married.
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
    # -------------------------------------------------------------------------
    # One-hot encoding:
    consolidated_df = pd.get_dummies(consolidated_df, columns=['belbezig_2020'], prefix='employment')    
    consolidated_df = pd.get_dummies(consolidated_df, columns=['partner_rec'], prefix='partner3cat')
    consolidated_df = pd.get_dummies(consolidated_df, columns=['oplcat_2020'], prefix='eduencoded')
    consolidated_df = pd.get_dummies(consolidated_df, columns=['woning_2020'], prefix='dwelling')
    consolidated_df = pd.get_dummies(consolidated_df, columns=['cr20m142'], prefix='religious')
    consolidated_df = pd.get_dummies(consolidated_df, columns=['cr20m143'], prefix='church')
    consolidated_df = pd.get_dummies(consolidated_df, columns=['migration_background_bg'], prefix='migration')
    consolidated_df = pd.get_dummies(consolidated_df, columns=['ch20m004'], prefix='health')
    consolidated_df = pd.get_dummies(consolidated_df, columns=['burgstat_2020'], prefix='civilstatus')
    consolidated_df = pd.get_dummies(consolidated_df, columns=['cf20m128'], prefix='morechildren')
    consolidated_df = pd.get_dummies(consolidated_df, columns=['getmarried_rec'], prefix='getmarried')
    consolidated_df = pd.get_dummies(consolidated_df, columns=['singleparent_rec'], prefix='singleparent')
    consolidated_df = pd.get_dummies(consolidated_df, columns=['parcontribute_rec'], prefix='parcontribute')
    # -------------------------------------------------------------------------
    # Partner recoded after one-hot enconding:
    consolidated_df.rename(columns={'civilstatus_1.0': 'civilstatus_married'}, inplace=True)
    # Create a new DataFrame for the merged column
    new_col = pd.DataFrame({
        'civilstatus_sepdivwid': (consolidated_df['civilstatus_2.0'] | consolidated_df['civilstatus_3.0'] | consolidated_df['civilstatus_4.0']).astype(int)
    })
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
       'health_3.0',
       'civilstatus_married',
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
       'partner3cat_1.0', # with partner
       'partner3cat_2.0', # non-resident partner
       'partner3cat_3.0', # no partner, leave 3 in case of singularity (3: low frequency)
       'cvresp' # Created by Abi
       ]
    #%% Create a new DataFrame with the selected columns
    df = consolidated_df[selected_features].copy()
    means = df.mean()
    # Replace NaNs with the mean of their respective columns in case of any missing
    df.isna().sum()
    df = df.fillna(means)
    df.isna().sum()
    return df

import joblib
def predict_outcomes(df, background_df=None, model_path="RFmodel_May11.joblib"):
    df = clean_df(df, background_df=None)
    ## This script contains a bare minimum working example
    if "nomem_encr" not in df.columns:
        print("The identifier variable 'nomem_encr' should be in the dataset")
    # Load the model
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