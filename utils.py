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
    df[cols] = df[cols].replace({'X': np.nan, 'XX':np.nan, '': np.nan, ' ':np.nan})
    df[cols] = df[cols].astype(float)
    
    return df

#function to check categorical variable counts
def categorical_checker(df, attributes_df):
    '''
    Takes in a feature dataframe and a demographic dataframe and prints the counts for categorical variables
    Args:
    df: demographics dataframe
    attributes_df: dataframe with the summary of all the features
    returns:
    nothing
    '''
    categorical = attributes_df[attributes_df['type'] == 'categorical']['attribute'].values
    categorical = [x for x in categorical if x in df.columns] 
    binary = [x for x in categorical if df[x].nunique()==2]
    multilevel = [x for x in categorical if df[x].nunique()>2]
    print(df[categorical].nunique())
    
#function to replace missing or unknow values based on the information in the attributes df
def feat_fixer(df, attributes_df):
    '''
    This function takes in any df and the attributes df and replaces missing and unknown values
    based on the information of the attributes dataframe
    Args:
    df: dany dataframe from this project with demo information
    attributes_df: dataframe with attributes summary
    returns:
    dataframe with missing and unknown replaced with nan
    '''
    df.replace({'X': np.nan, 'XX':np.nan, '': np.nan, ' ':np.nan})
    #parsing unknown and missing values from the attributes dataframe
    m_o_u_list = [x.replace("[","").replace("]","").split(',') for x in attributes_df['missing_or_unknown']]
    
    #changing strings to floats
    m_o_u_float = []
    for x in m_o_u_list:
        #list inside list
        list_of_list = []
        for missing in x:
            try:
                missing = float(missing)
                list_of_list.append(missing)
            except:
                missing = np.nan
                list_of_list.append(missing)
        list_of_list
                
        m_o_u_float.append(list_of_list)
        
        #replacing the missing and unknown values with nan
    for col, m_unknown in zip(df.columns, m_o_u_float):
        for miss in m_unknown:
            df[col].replace(m_unknown, np.nan, inplace = True)
        
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