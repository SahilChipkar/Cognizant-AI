import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

def load_data(path: str = "/path/to/csv/"):
    """
    Load a CSV file into a Pandas DataFrame.

    :param path: str, relative path of the CSV file
    :return: pd.DataFrame
    """
    df = pd.read_csv(path)
    df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
    return df

def split_data(data: pd.DataFrame = None, target: str = "estimated_stock_pct"):
    """
    Split the columns of a DataFrame into predictor variables (X) and a target variable (y).

    :param data: pd.DataFrame, dataframe containing data for the model
    :param target: str, target variable to predict
    :return: X: pd.DataFrame, y: pd.Series
    """
    if target not in data.columns:
        raise Exception(f"Target variable '{target}' is not present in the data.")
    X = data.drop(columns=[target])
    y = data[target]
    return X, y

def train_model_with_cross_validation(X: pd.DataFrame = None, y: pd.Series = None, num_folds: int = 5):
    """
    Train a Random Forest Regressor model using K-fold cross-validation.

    :param X: pd.DataFrame, predictor variables
    :param y: pd.Series, target variable
    :param num_folds: int, number of folds for cross-validation
    """
    accuracy = []
    for fold in range(num_folds):
        model = RandomForestRegressor()
        scaler = StandardScaler()

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        trained_model = model.fit(X_train_scaled, y_train)

        y_pred = trained_model.predict(X_test_scaled)

        mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
        accuracy.append(mae)
        print(f"Fold {fold + 1}: MAE = {mae:.3f}")

    average_mae = sum(accuracy) / len(accuracy)
    print(f"Average MAE: {average_mae:.2f}")
