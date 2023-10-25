## IMPORTS ##

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import scipy as sp
from pydataset import data
from env import user, password, host

import warnings
warnings.filterwarnings("ignore")

import os
directory = os.getcwd()
seed = 3333


## FUNCTIONS ##
##-------------------------------------------------------------------##

def get_db_url(database_name):
    """
    this function will:
    - take in a string database_name 
    - return a string connection url to be used with sqlalchemy later.
    """
    return f'mysql+pymysql://{user}:{password}@{host}/{database_name}'

def new_telco_data():
    """
    This function will:
    - take in a SQL_query
    - create a connection_url to mySQL
    - return a df of the given query from the telco_db
    """
    
    sql_query = """
        SELECT * FROM customers
        JOIN contract_types USING (contract_type_id)
        JOIN internet_service_types USING (internet_service_type_id)
        JOIN payment_types USING (payment_type_id)
        """
    
    url = get_db_url('telco_churn')
    
    df = pd.read_sql(sql_query, url)
    
    return df

def get_telco_data():
    """
    This function will:
    - Check local directory for csv file
        - return if exists
    - if csv doesn't exist:
        - creates df of sql query
        - writes df to csv
    - outputs telco df
    """
    filename = 'telco.csv'
    
    if os.path.isfile(filename): 
        df = pd.read_csv(filename)
        return df
    else:
        df = new_telco_data()

        df.to_csv(filename)
    return df
    
    

def split_data_telco(df):
    seed = 3333
    train, test = train_test_split(df,
                               train_size = 0.8,
                               stratify = df.churn,
                               random_state=seed)
    train, validate = train_test_split(train,
                                  train_size = 0.75,
                                  stratify = train.churn,
                                  random_state=seed)
    return train, validate, test

def split_data_telco2(df):
    seed = 3333
    train, test = train_test_split(df,
                               train_size = 0.8,
                               stratify = df.churn_encoded,
                               random_state=seed)
    train, validate = train_test_split(train,
                                  train_size = 0.75,
                                  stratify = train.churn_encoded,
                                  random_state=seed)
    return train, validate, test



def prep_telco_data(df):
    # Ensure necessary libraries are imported
    import pandas as pd
    
    # Drop duplicate columns
    cols_to_drop = ['Unnamed:0', 'payment_type_id', 'internet_service_type_id', 'contract_type_id', 'customer_id']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

    
    # Drop null values stored as whitespace    
    df = df[df.total_charges != '']
    
    # Convert to correct datatype
    df['total_charges'] = df.total_charges.astype(float)
    # Use df instead of telco
    df['avg_tenure_charges'] = (df['total_charges'] / df['tenure']).round(2)
    
    # Convert binary categorical variables to numeric
    df['gender_encoded'] = df.gender.map({'Female': 1, 'Male': 0})
    df['partner_encoded'] = df.partner.map({'Yes': 1, 'No': 0})
    df['dependents_encoded'] = df.dependents.map({'Yes': 1, 'No': 0})
    df['phone_service_encoded'] = df.phone_service.map({'Yes': 1, 'No': 0})
    df['paperless_billing_encoded'] = df.paperless_billing.map({'Yes': 1, 'No': 0})
    df['churn_encoded'] = df.churn.map({'Yes': 1, 'No': 0})
    
    # Get dummies for non-binary categorical variables
    dummy_df = pd.get_dummies(df[['multiple_lines', 'online_security', 'online_backup', 
                                  'device_protection', 'tech_support', 'streaming_tv', 
                                  'streaming_movies', 'contract_type', 'internet_service_type', 
                                  'payment_type']], dummy_na=False, drop_first=True)
    
    # Concatenate dummy dataframe to original 
    df = pd.concat([df, dummy_df], axis=1)
    # Drop columns replaced with dummy_df.
    df = df.drop(columns=['gender', 'partner', 'dependents', 'phone_service', 'paperless_billing', 'churn','multiple_lines', 'online_security', 'online_backup', 'device_protection','tech_support', 'streaming_tv', 'streaming_movies', 'contract_type', 'internet_service_type', 'payment_type'])
    
    # Ensure the split_data_telco function is available
    train, validate, test = split_data_telco2(df)
    
    return train, validate, test
