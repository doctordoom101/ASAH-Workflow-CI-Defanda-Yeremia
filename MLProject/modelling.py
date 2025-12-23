import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
import os

def train():
    # Menerima argument dari MLProject
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="housevalue_preprocessing")
    args = parser.parse_args()

    # Load data (dinamis berdasarkan path)
    X_train = pd.read_csv(os.path.join(args.data_path, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(args.data_path, "y_train.csv"))

    # Tracking
    with mlflow.start_run():
        model = LinearRegression()
        model.fit(X_train, y_train.values.ravel())
        
        # Log model & artefak
        mlflow.sklearn.log_model(model, "model")
        print("Training selesai dalam MLflow Project environment.")

if __name__ == "__main__":
    train()