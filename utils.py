import numpy as np
import pandas as pd
import collections

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

# function to determine if 2 dataframes are balanced in terms of number and type of features
def balance_checker(df1, df2):
    '''
    Takes in 2 dataframes and checks if attributes match
    '''
    features_list_df1 = df1.columns.values
    features_list_df2 = df2.columns.values
    equal = collections.Counter(features_list_df1) == collections.Counter(features_list_df2)
    
    print('Feature balance between dfs?: ', equal)
    
    if equal == False:
        print('Your first argument df differs from the second on the following columns: ')
        print(set(features_list_df1) - set(features_list_df2))
        
        print('Your second argument df differs from the first on the following columns: ')
        print(set(features_list_df2) - set(features_list_df1))
        
        
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

#function to delete columns with too much missing data
def columns_to_delete(df):
    '''
    Fuction goes through dataframe created with split_on_percentage() and saves column names over
    the chosen boundary and saves column names to a list
    Args: dataframe created using split_on_percentage
    returns: list of the columns we want to exclude
    '''
    cols_del = df.index.values.tolist()
    
    return cols_del

#function to delete rows with too much missing data
def row_dropper(df, boundary):
    '''
    This function identifies rows missing more than a threshold amount of data specified with boundary
    and drops them
    Args: 
    dataframe: already cleaned up of columns missing more than a boundary defined percentage 
    boundary: number of missing entries limit to droppable rows
    
    returns:
    dataframe with dropped rows with more than a percentage of missing values
    '''
    df = df.dropna(thresh=df.shape[1]-boundary)
    df = df.reset_index()
    del df['index']
    
    return df

#function to handle special feature columns
def special_feature_handler(df):
    #extract the time,and keep the year for column with date/time information
    df["EINGEFUEGT_AM"]=pd.to_datetime(df["EINGEFUEGT_AM"]).dt.year
    
    #OST_WEST_KZ is a binary feature that needs encoding it takes the values array(['W', 'O'], dtype=object)
    o_w_k_dict = {'OST_WEST_KZ': {'W':0, 'O':1}}
    df = df.replace(o_w_k_dict)
    
    return df

