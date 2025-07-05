import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import TargetEncoder, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

def load_dataset(path):
    '''
    Arg: path of csv file
    Loads data from csv file to dataframe
    Filters data to only 'Spain'
    Selects only columns to work with for the model
    Removes nas from target columns "Price"
    Returns: Selected Dataframe based on EDA done previously
    '''
    df_airbnb = pd.read_csv(path, delimiter=";")
    df_airbnb = df_airbnb[df_airbnb['Country'] == 'Spain']
    desired_columns = ['Neighbourhood Cleansed', 'Smart Location', 'Latitude', 'Longitude', 'Property Type', 'Bathrooms',
    'Bedrooms', 'Beds', 'Guests Included', 'Extra People', 'Minimum Nights',
    'Maximum Nights', 'Number of Reviews', 'Room Type', 'Price']
    df_airbnb = df_airbnb[desired_columns]
    df_airbnb.dropna(subset=['Price'], inplace=True)

    return df_airbnb

def create_train_test(df):
    """
    Takes selected dataframe and splits it into train and test subsets.
    Returns dataframes of x_train and x_test and arrays with y_train and y_test with the target 'Price'
    """
    # Split data into train and test sets
    train, test = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)
    y_test = test['Price']
    test[['Price']].to_csv('test-target.csv', index=False)
    x_test = test.drop(['Price'], axis=1)
    test.to_csv('test.csv', index=False)

    features = [x for x in list(train.columns) if x != 'Price']
    x_train = train[features]
    y_train = train['Price']

    y_train = np.log1p(y_train) # model use log(price)
    y_test = np.log1p(y_test)

    return x_train, x_test, y_train, y_test


def preprocess_data_train(x_train, y_train):
    """
    Takes train subset and makes all preprocess operations to feed it into the model
    Transforms categoric columns and Inputs NAs of some columns
    Returns: transformed x_train, encoders for oneHot and TargetEncoder
    """
    # Procesamiento de valores nulos
    # Para columnas numéricas
    for col in ['Bedrooms', 'Bathrooms', 'Beds']:
        x_train[col] = x_train[col].fillna(x_train[col].mean())

    # Target Encoder for 'Neighbourhood Cleansed', 'Smart Location', 'Property Type'.
    categorical_cols = ['Neighbourhood Cleansed', 'Smart Location', 'Property Type']
    encoder_target = TargetEncoder()
    encoded = encoder_target.fit_transform(x_train[categorical_cols], y_train)
    x_train[categorical_cols] = encoded

    # OneHot Encoding to 'Room_Type'
    encoder_one = OneHotEncoder(
        sparse_output=False,  # Devuelve un array denso (no una matriz sparse)
        drop='first',        # Opcional: elimina una categoría para evitar multicolinealidad
        handle_unknown='ignore'  # Ignora categorías nuevas en test (opcional pero recomendado)
    )
    ohe_result = encoder_one.fit_transform(x_train[['Room Type']])
    ohe_columns = encoder_one.get_feature_names_out(['Room Type'])  # Nombres de las columnas
    ohe_df = pd.DataFrame(ohe_result, columns=ohe_columns, index=x_train.index)
    x_train = pd.concat([x_train, ohe_df], axis=1)
    x_train = x_train.drop(['Room Type'], axis=1)

    return x_train, encoder_target, encoder_one

def preprocess_data_test(x_test, x_train, y_train, encoder_target, encoder_one):
    '''
    Takes test subset and makes all preprocess operations to feed it into the model
    Transforms categoric columns and Inputs NAs of some columns using training means and trained encoders
    Returns: transformed x_test
    '''
     # Fill NA for numerical columns using training means
    for col in ['Bedrooms', 'Bathrooms', 'Beds']:
        x_test[col] = x_test[col].fillna(x_train[col].mean())

    # Apply target encoder (fitted on x_train)
    categorical_cols = ['Neighbourhood Cleansed', 'Smart Location', 'Property Type']
    encoded = encoder_target.transform(x_test[categorical_cols])
    x_test[categorical_cols] = encoded

    # One-Hot Encoding using the same columns as training
    ohe_result = encoder_one.transform(x_test[['Room Type']])
    ohe_columns = encoder_one.get_feature_names_out(['Room Type']) 
    ohe_df = pd.DataFrame(ohe_result, columns=ohe_columns, index=x_test.index)
    x_test = pd.concat([x_test.drop(['Room Type'], axis=1), ohe_df], axis=1)

    return x_test

def Scale_numeric_var(x_train, x_test):
    """
    Transforms with StandardScaler those numeric columns in the X matrices of train and test
    """
    #1. Separate binary and categoric variables
    X_train_numeric = x_train.iloc[:,:-2]
    X_train_binary = x_train.iloc[:, -2]

    X_test_numeric = x_test.iloc[:,:-2]
    X_test_binary = x_test.iloc[:, -2]

    #2. Apply Scaler on numeric data
    scaler = StandardScaler().fit(X_train_numeric)
    Xtrain_numeric_scaled = scaler.transform(X_train_numeric)  
    Xtest_numeric_scaled = scaler.transform(X_test_numeric)

    #3. Join scaled data with binary columns
    Xtrain_numeric_scaled = pd.DataFrame(Xtrain_numeric_scaled, columns=X_train_numeric.columns, index=X_train_numeric.index)
    Xtrain_final = pd.concat([Xtrain_numeric_scaled, X_train_binary], axis=1)

    Xtest_numeric_scaled = pd.DataFrame(Xtest_numeric_scaled, columns=X_test_numeric.columns, index=X_test_numeric.index)
    Xtest_final = pd.concat([Xtest_numeric_scaled, X_test_binary], axis=1)

    return Xtrain_final, Xtest_final

def mlflow_tracking(nombre_job, x_train, x_test, y_train, y_test, n_estimators, max_depth, min_samples_leaf):
    '''
    Trains and registers results from trainings in Mlflow for different values of three hiperparameters for a Random Forest Model.
    Args:
        name of the experiment,
        x_train, x_test, y_train, y_test,
        Hiperparameters: n_estimators, max_depth, min_samples_leaf: Need to be tuples of a range of values to test in the experiments
    '''
    time.sleep(5)
    
    if mlflow.active_run():
        mlflow.end_run()
    mlflow.set_experiment(nombre_job)
    for i in range(*n_estimators):
        for u in range(*max_depth):
            for e in range(*min_samples_leaf):
                with mlflow.start_run(run_name=f"RF_{i}_estimators_{u}_depth_sample_leaf{e}"):
                    model = RandomForestRegressor(n_estimators=i,
                                                min_samples_leaf=e,
                                                random_state=123,
                                                max_depth=u
                                                )

                    #preprocessor = Pipeline(steps=[('scaler', StandardScaler())])

                    model.fit(x_train, y_train)
                    r2_train = model.score(x_train, y_train) 
                    r2_test = model.score(x_test, y_test)    

                    mlflow.log_metric('r2_train', r2_train)
                    mlflow.log_metric('r2_test', r2_test)
                    mlflow.log_param('n_estimators', i)
                    mlflow.log_param('max_depth', u)
                    mlflow.log_param('min_samples_leaf', e)
                    mlflow.sklearn.log_model(model, artifact_path="random_forest_model")
            
    print("Training succesfully finished")
