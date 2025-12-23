import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
import os

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="housevalue_preprocessing")
    args = parser.parse_args()

    # Load data
    X_train = pd.read_csv(os.path.join(args.data_path, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(args.data_path, "y_train.csv"))

    # GUNAKAN NESTED START_RUN ATAU CEK ACTIVE RUN
    # Ini krusial untuk mencegah error "Run not found" di MLflow Project
    active_run = mlflow.active_run()
    
    with (active_run if active_run else mlflow.start_run()):
        model = LinearRegression()
        model.fit(X_train, y_train.values.ravel())
        
        mlflow.sklearn.log_model(model, "model")
        print("Training selesai dengan sukses.")

if __name__ == "__main__":
    train()