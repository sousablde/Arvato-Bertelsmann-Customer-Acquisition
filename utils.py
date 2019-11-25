import numpy as np
import pandas as pd

#function to fix mixed type columns
def mixed_type_fixer(df, cols):
    '''
    Function to convert mixed type columns to floats
    ARGS:
    dataframe: df we need to fix
    cols: cols in need to be fixed
    returns:
    dataframe with fixed columns
    '''
    df[cols] = df[cols].replace({'X': np.nan, 'XX':np.nan})
    df[cols] = df[cols].astype(float)
    
    return df

# creating a function to determine percentage of missing values
def percentage_of_missing(df):
    '''
    This function calculates the percentage of missing values in a dataframe and splits it on a defined
    percentage boundary
    inputs: dataframe
    output: missing values dataframe
    '''
    percent_missing = df.isnull().sum()* 100/len(df)
    percent_missing_df = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': percent_missing})
    return percent_missing_df


#creating a function to split a percentage of missing values dataframe for visualization purposes
def split_on_percentage(df, boundary, conditional):
    '''
    This function takes in a dataframe and splits it on a particular percentage boundary with a particular conditional
    Args: dataframe created with percentage_of_missing
    boundary: percentage value we want to upper bound or lower bound values for
    conditional: determines if we are getting greater or less than values
    '''
    if conditional == '>':
        split_df = df[df.percent_missing > boundary]
    elif conditional == '>=':
        split_df = df[df.percent_missing >= boundary]
    elif conditional == '<=':
        split_df = df[df.percent_missing <= boundary]
    else:
        split_df = df[df.percent_missing > boundary]
    
    return split_df  

def columns_to_delete(df):
    '''
    Fuction goes through dataframe created with split_on_percentage() and saves column names over
    the chosen boundary and saves column names to a list
    Args: dataframe created using split_on_percentage
    returns: list of the columns we want to exclude
    '''
    cols_del = df.index.values.tolist()
    
    return cols_del