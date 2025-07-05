from functions import *

def main(path, experiment_name='defaut_name', n_estimators:tuple=(100, 100), max_depth:tuple=(20,20), min_samples_leaf:tuple=(4,5)): 

    df = load_dataset(path)

    x_train, x_test, y_train, y_test = create_train_test(df)

    x_train, tar_encoder, one_encoder = preprocess_data_train(x_train, y_train)
    x_test = preprocess_data_test(x_test, x_train, y_train, tar_encoder, one_encoder)

    x_train_scaled, x_test_scaled = Scale_numeric_var(x_train, x_test)

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow_tracking(experiment_name, x_train, x_test, y_train, y_test, n_estimators, max_depth, min_samples_leaf)

if __name__ == '__main__': 
    main()