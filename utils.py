import numpy as np
import pandas as pd
import collections
import matplotlib.pyplot as plt
from pylab import *

# sklearn
from sklearn.preprocessing import StandardScaler, Imputer, RobustScaler, MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels


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

#function to plot/visualize histogram of data missing in rows
def row_hist(df1, df2, bins):
    '''
    This function takes in the azdias and the customers dataframe, and the number of bins
    we want the data to be distributed by and plots a histogram of nulls distribution accross
    rows
    '''
   
    plt.hist(df1.isnull().sum(axis=1), bins, color = 'orange')

    plt.hist(df2.isnull().sum(axis=1), bins, color = 'green')

    plt.title('Distributions of null values in Azdias and Customers rows')
    plt.xlabel('# Null Values')
    plt.ylabel('Rows')

    plt.show()
    

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

#function for features engineering: creating novel features
def feat_eng(df):
    
    #creating the dictionaries for mapping in PRAEGENDE_JUGENDJAHRE
    #decades:
    decades_dict = {1: 40, 2: 40, 3: 50, 4: 50, 5: 60, 6: 60, 7: 60,
           8: 70, 9: 70, 10: 80, 11: 80, 12: 80, 13: 80, 14: 90,
           15: 90, 0: np.nan}
    df['PRAEGENDE_JUGENDJAHRE_DECADE'] = df['PRAEGENDE_JUGENDJAHRE'].map(decades_dict)
    #mainstream or avant-garde movement
    movement_dict = {1: 0, 2: 1, 3: 0, 4: 1, 5: 0, 6: 1, 7: 1, 8: 0,
           9: 1, 10: 0, 11: 1, 12: 0, 13: 1, 14: 0, 15: 1, 0: np.nan}
    df['PRAEGENDE_JUGENDJAHRE_MOVEMENT'] = df['PRAEGENDE_JUGENDJAHRE'].map(movement_dict)

       
    # WOHNLAGE refers to neighborhood area, from very good to poor; rural
    #creating dictionaries for WOHNLAGE
    area_dict = {1.0:0, 2.0:0, 3.0:0, 4.0:0, 5.0:0, 7.0:1, 8.0:1}
    #creating a feature for borough quality
    df['WOHNLAGE_QUALITY'] = df[(df['WOHNLAGE'] > 0) & (df['WOHNLAGE'] < 7)]['WOHNLAGE']
    #creating a feature for rural/urban division
    df['WOHNLAGE_AREA'] = df['WOHNLAGE'].map(area_dict)
    
    #Using CAMEO to create a wealth and family type feature
    df['WEALTH'] = df['CAMEO_INTL_2015'].apply(lambda x: np.floor_divide(float(x), 10) if float(x) else np.nan)
    df['FAMILY'] = df['CAMEO_INTL_2015'].apply(lambda x: np.mod(float(x), 10) if float(x) else np.nan)
    
    #dealing with LP_LEBENSPHASE_FEIN
    life_stage = {1: 'younger_age', 2: 'middle_age', 3: 'younger_age',
              4: 'middle_age', 5: 'advanced_age', 6: 'retirement_age',
              7: 'advanced_age', 8: 'retirement_age', 9: 'middle_age',
              10: 'middle_age', 11: 'advanced_age', 12: 'retirement_age',
              13: 'advanced_age', 14: 'younger_age', 15: 'advanced_age',
              16: 'advanced_age', 17: 'middle_age', 18: 'younger_age',
              19: 'advanced_age', 20: 'advanced_age', 21: 'middle_age',
              22: 'middle_age', 23: 'middle_age', 24: 'middle_age',
              25: 'middle_age', 26: 'middle_age', 27: 'middle_age',
              28: 'middle_age', 29: 'younger_age', 30: 'younger_age',
              31: 'advanced_age', 32: 'advanced_age', 33: 'younger_age',
              34: 'younger_age', 35: 'younger_age', 36: 'advanced_age',
              37: 'advanced_age', 38: 'retirement_age', 39: 'middle_age',
              40: 'retirement_age'}

    fine_scale = {1: 'low', 2: 'low', 3: 'average', 4: 'average', 5: 'low', 6: 'low',
              7: 'average', 8: 'average', 9: 'average', 10: 'wealthy', 11: 'average',
              12: 'average', 13: 'top', 14: 'average', 15: 'low', 16: 'average',
              17: 'average', 18: 'wealthy', 19: 'wealthy', 20: 'top', 21: 'low',
              22: 'average', 23: 'wealthy', 24: 'low', 25: 'average', 26: 'average',
              27: 'average', 28: 'top', 29: 'low', 30: 'average', 31: 'low',
              32: 'average', 33: 'average', 34: 'average', 35: 'top', 36: 'average',
              37: 'average', 38: 'average', 39: 'top', 40: 'top'}
    
    df['LP_LEBENSPHASE_FEIN_life_stage'] = df['LP_LEBENSPHASE_FEIN'].map(life_stage)
    df['LP_LEBENSPHASE_FEIN_fine_scale'] = df['LP_LEBENSPHASE_FEIN'].map(fine_scale)
    
    life_dict = {'younger_age': 1, 'middle_age': 2, 'advanced_age': 3,
            'retirement_age': 4}
    scale_dict = {'low': 1, 'average': 2, 'wealthy': 3, 'top': 4}

    df['LP_LEBENSPHASE_FEIN_life_stage'] = df['LP_LEBENSPHASE_FEIN_life_stage'].map(life_dict)
    df['LP_LEBENSPHASE_FEIN_fine_scale'] = df['LP_LEBENSPHASE_FEIN_fine_scale'].map(scale_dict)
    
    
    #dropping columns used to create new features, have object types or duplicated information (ie. grob/fein)
    cols = ['PRAEGENDE_JUGENDJAHRE', 'WOHNLAGE', 'CAMEO_INTL_2015','LP_LEBENSPHASE_GROB', 'LP_LEBENSPHASE_FEIN',
            'D19_LETZTER_KAUF_BRANCHE' ]
    df.drop(cols, axis = 1, inplace = True)
    
    
    return df

#pca model
def pca_model(df, n_components):
    '''
    This function defines a model that takes in a previously scaled dataframe and returns the result of 
    the transformation. The output is an onject created post data fitting
    '''
    pca = PCA(n_components)
    pca_df = pca.fit(df)
    
    return pca_df

#scree plots for PCA
def scree_plots(SS, RS, MMS, dataname):
    '''
    This function takes in the transformed data using PCA and plots it in scree plots
    '''
    subplot(3,1,1)

    plt.plot(np.cumsum(SS.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance Ratio vs Number of Components SS' + dataname)
    plt.grid(b=True)



    subplot(3,1,2)
    plt.plot(np.cumsum(RS.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance Ratio vs Number of Components RS' + dataname)
    plt.grid(b=True)


    subplot(3,1,3)
    plt.plot(np.cumsum(MMS.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance Ratio vs Number of Components MMS' + dataname)
    plt.grid(b=True)

    plot = tight_layout()
    plot = plt.show()

