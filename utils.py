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
    cameo_dict = {'1A':1, '1B':2, '1C': 3, '1D':4, '1E':5,
                 '2A':6, '2B':7, '2C': 8, '2D':9,
                 '3A':10, '3B':11, '3C': 12, '3D':13,
                 '4A':14, '4B':15, '4C': 16, '4D':17, '4E':18,
                 '5A':19, '5B':20, '5C': 21, '5D':22, '5E':23, '5F': 24,
                 '6A':25, '6B':26, '6C': 27, '6D':28, '6E':29, '6F': 30,
                 '7A':31, '7B':32, '7C': 33, '7D':34, '7E':35,
                 '8A':36, '8B':37, '8C': 38, '8D':39,
                 '9A':40, '9B':41, '9C': 42, '9D':43, '9E':44,}
    df['CAMEO_DEU_2015'] = df['CAMEO_DEU_2015'].map(cameo_dict)
    df[cols] = df[cols].replace({'X': np.nan, 'XX':np.nan, '': np.nan, ' ':np.nan})
    df[cols] = df[cols].astype(float)
    
    #drop the unnamed column
    df = df.drop(df.columns[0], axis = 1)
    
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
        
#function to deal with all the missing and unknown entries
def unknowns_to_NANs(df, xls):
    #using the DIAs xls file lets save meanings that might indicate unknown values
    unknowns = xls['Meaning'].where(xls['Meaning'].str.contains('unknown')).value_counts().index
    
    #I will now create a list of all the unknown values for each attribute and replace them on my azdias and customers
    missing_unknowns = xls[xls['Meaning'].isin(unknowns)]
    
    for row in missing_unknowns.iterrows():
        missing_values = row[1]['Value']
        attribute = row[1]['Attribute']
        
        #dealing with columns that only exist in df
        if attribute not in df.columns:
            continue
        
        #dealing with strings or ints
        if isinstance(missing_values,int): 
            df[attribute].replace(missing_values, np.nan, inplace=True)
        elif isinstance(missing_values,str):
            eval("df[attribute].replace(["+missing_values+"], np.nan, inplace=True)")
        
        
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
    print('Creating PRAEGENDE_JUGENDJAHRE_DECADE feature')
    
    #mainstream or avant-garde movement
    movement_dict = {1: 0, 2: 1, 3: 0, 4: 1, 5: 0, 6: 1, 7: 1, 8: 0,
           9: 1, 10: 0, 11: 1, 12: 0, 13: 1, 14: 0, 15: 1, 0: np.nan}
    df['PRAEGENDE_JUGENDJAHRE_MOVEMENT'] = df['PRAEGENDE_JUGENDJAHRE'].map(movement_dict)
    
    print('Creating PRAEGENDE_JUGENDJAHRE_MOVEMENT feature')
       
    # WOHNLAGE refers to neighborhood area, from very good to poor; rural
    #creating dictionaries for WOHNLAGE
    area_dict = {1.0:0, 2.0:0, 3.0:0, 4.0:0, 5.0:0, 7.0:1, 8.0:1}
    #creating a feature for borough quality
    df['WOHNLAGE_QUALITY'] = df[(df['WOHNLAGE'] > 0) & (df['WOHNLAGE'] < 7)]['WOHNLAGE']
    
    print('Creating WOHNLAGE_QUALITY feature')
    
    #creating a feature for rural/urban division
    df['WOHNLAGE_AREA'] = df['WOHNLAGE'].map(area_dict)
    print('Creating WOHNLAGE_AREA feature')
    
    
    #Using CAMEO to create a wealth and family type feature
    df['WEALTH'] = df['CAMEO_INTL_2015'].apply(lambda x: np.floor_divide(float(x), 10) if float(x) else np.nan)
    df['FAMILY'] = df['CAMEO_INTL_2015'].apply(lambda x: np.mod(float(x), 10) if float(x) else np.nan)
    print('Creating Wealth and Family feature')
    
    
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
    
    print('Creating LP_LEBENSPHASE_FEIN_life_stage and LP_LEBENSPHASE_FEIN_fine_scale feature')
    
    #I could not find much information on D19_LETZTER_KAUF_BRANCHE so I will have an agnostic approach to it
    branch = {'D19_UNBEKANNT':1, 'D19_SCHUHE':2, 'D19_ENERGIE':3, 'D19_KOSMETIK':4,
       'D19_VOLLSORTIMENT':5, 'D19_SONSTIGE':6, 'D19_BANKEN_GROSS':7,
       'D19_DROGERIEARTIKEL':8, 'D19_HANDWERK':9, 'D19_BUCH_CD':10,
       'D19_VERSICHERUNGEN':11, 'D19_VERSAND_REST':12, 'D19_TELKO_REST':13,
       'D19_BANKEN_DIREKT':14, 'D19_BANKEN_REST':15, 'D19_FREIZEIT':16,
       'D19_LEBENSMITTEL':17, 'D19_HAUS_DEKO':18, 'D19_BEKLEIDUNG_REST':19,
       'D19_SAMMELARTIKEL':20, 'D19_TELKO_MOBILE':21, 'D19_REISEN':22,
       'D19_BEKLEIDUNG_GEH':23, 'D19_TECHNIK':24, 'D19_NAHRUNGSERGAENZUNG':25,
       'D19_DIGIT_SERV':26, 'D19_LOTTO':27, 'D19_RATGEBER':28, 'D19_TIERARTIKEL':29,
       'D19_KINDERARTIKEL':30, 'D19_BIO_OEKO':31, 'D19_WEIN_FEINKOST':32,
       'D19_GARTEN':33, 'D19_BILDUNG':34, 'D19_BANKEN_LOKAL':35}
    
    df['D19_LETZTER_KAUF_BRANCHE_NUM'] = df['D19_LETZTER_KAUF_BRANCHE'].map(branch)
    print('Creating D19_LETZTER_KAUF_BRANCHE feature')
    
    #Creating Nationality
    nat = {0: np.nan, 1:1, 2:2, 3:3, 4:4}
    df['NATIONALITY'] = df['NATIONALITAET_KZ'].map(nat)
    print('Creating NATIONALITY feature')
    

    
    #dropping columns used to create new features, have object types or duplicated information (ie. grob/fein)
    cols = ['PRAEGENDE_JUGENDJAHRE', 'WOHNLAGE', 'CAMEO_INTL_2015','LP_LEBENSPHASE_GROB', 'LP_LEBENSPHASE_FEIN',
            'D19_LETZTER_KAUF_BRANCHE', 'NATIONALITAET_KZ' ]
    df.drop(cols, axis = 1, inplace = True)
    
    
    return df
#function to scale and normalize the dataframes features
def feature_scaling(df, type_scale):
    
    features_list = df.columns
    
    #dealing with remaining missing values
    df.fillna(0, inplace = True)
    
    if type_scale == 'StandardScaler':
        df_scaled = StandardScaler().fit_transform(df)
        
    if type_scale == 'RobustScaler':
        df_scaled = RobustScaler().fit_transform(df)
        
    if type_scale == 'MinMaxScaler':
        df_scaled = MinMaxScaler().fit_transform(df)
    
    df_scaled = pd.DataFrame(df_scaled)
    df_scaled.columns = features_list
    
    return df_scaled

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

#function to help interpret the pca results
def interpret_pca(df, n_components, component):
    '''
    Maps each weight to its corresponding feature name and sorts according to weight.
    Args:
        df (dataframe): dataframe on which pca has been used on.
        pca (pca): pca object.
        component (int): which principal compenent to return
    Returns:
        df_pca (dataframe): dataframe for specified component containing the explained variance
                            and all features and weights sorted according to weight.
    '''
    pca = PCA(n_components)
    df_pca = pca.fit_transform(df)
    
    df_pca = pd.DataFrame(columns=list(df.columns))
    df_pca.loc[0] = pca.components_[component]
    dim_index = "Dimension: {}".format(component + 1)

    df_pca.index = [dim_index]
    df_pca = df_pca.loc[:, df_pca.max().sort_values(ascending=False).index]

    ratio = np.round(pca.explained_variance_ratio_[component], 4)
    df_pca['Explained Variance'] = ratio

    cols = list(df_pca.columns)
    cols = cols[-1:] + cols[:-1]
    df_pca = df_pca[cols]

    return df_pca

#function to display interesting features
def display_interesting_features(df, pca, dimensions):
    
    features = df.columns.values
    components = pca.components_
    feature_weights = dict(zip(features, components[dimensions]))
    sorted_weights = sorted(feature_weights.items(), key = lambda kv: kv[1])
    
    print('Lowest: ')
    for feature, weight, in sorted_weights[:3]:
        print('\t{:20} {:.3f}'.format(feature, weight))
    
    print('Highest: ')
    for feature, weight in sorted_weights[-3:]:
        print('\t{:20} {:.3f}'.format(feature, weight))
        
#function to fit the kmeans model
def fit_kmeans(data, centers):
    '''
    returns the kmeans score regarding SSE for points to centers
    INPUT:
        data - the dataset you want to fit kmeans to
        center - the number of centers you want (the k value)
    OUTPUT:
        score - the SSE score for the kmeans model fit to the data
    
    '''
    kmeans = KMeans(centers)
    model = kmeans.fit(data)
    # SSE score for kmeans model 
    score = np.abs(model.score(data))
    return score
        
        
#function to dispplay elbow plot
def elbow_method(data):
    scores = []
    centers = list(range(1,20))
    i = 0
    for center in centers:
        i += 1
        print(i)
        scores.append(fit_kmeans(data, center))
        
    # Investigate the change in within-cluster distance across number of clusters.
    # Plot the original data with clusters
    plt.plot(centers, scores, linestyle='--', marker='o', color='b')
    plt.ylabel('SSE score')
    plt.xlabel('K')
    plt.title('SSE vs K')

    #Using a regression to determine where it is a good cluster number to divide the population (when the gradient decreases)
    l_reg = LinearRegression()
    l_reg.fit(X=np.asarray([[9,10,11,12,13,14]]).reshape(6,1), y=scores[8:14])
    predicted =l_reg.predict(np.asarray(range(2,9)).reshape(-1,1))
    plt.plot(list(range(2,20)),np.asarray(list(predicted.reshape(-1,1)) + list(scores[8:20])),'r')