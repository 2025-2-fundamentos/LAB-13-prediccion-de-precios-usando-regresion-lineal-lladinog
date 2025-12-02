#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#

import os
import json
import gzip
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error


def read_data():
    train_df = pd.read_csv("files/input/train_data.csv.zip", compression="zip")
    test_df = pd.read_csv("files/input/test_data.csv.zip", compression="zip")
    return train_df, test_df


def engineer_features(df):
    df = df.copy()
    df["Age"] = 2025 - df["Year"]
    df.drop(columns=["Year", "Car_Name"], inplace=True)
    return df


def split_xy(df):
    X = df.drop(columns=["Present_Price"])
    y = df["Present_Price"]
    return X, y


def build_pipeline(X):
    categorical = ["Fuel_Type", "Selling_type", "Transmission"]
    numerical = [col for col in X.columns if col not in categorical]
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(), categorical),
            ("scaler", MinMaxScaler(), numerical),
        ]
    )
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("feature_selection", SelectKBest(f_regression)),
        ("regressor", LinearRegression()),
    ])
    return pipeline


def train_model(pipeline, X, y):
    params = {
        "feature_selection__k": range(1, 12),
        "regressor__fit_intercept": [True, False],
        "regressor__positive": [True, False]
    }
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=params,
        cv=10,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        refit=True,
        verbose=1
    )
    grid.fit(X, y)
    return grid


def save_model(model):
    os.makedirs("files/models/", exist_ok=True)
    with gzip.open("files/models/model.pkl.gz", "wb") as f:
        pickle.dump(model, f)


def compute_metrics(y_true, y_pred, split):
    return {
        "type": "metrics",
        "dataset": split,
        "r2": float(r2_score(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mad": float(median_absolute_error(y_true, y_pred)),
    }


def save_metrics(metrics):
    os.makedirs("files/output/", exist_ok=True)
    with open("files/output/metrics.json", "w") as f:
        for m in metrics:
            f.write(json.dumps(m) + "\n")


def main():
    train_df, test_df = read_data()
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)
    X_train, y_train = split_xy(train_df)
    X_test, y_test = split_xy(test_df)
    pipeline = build_pipeline(X_train)
    model = train_model(pipeline, X_train, y_train)
    save_model(model)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    metrics = [
        compute_metrics(y_train, y_train_pred, "train"),
        compute_metrics(y_test, y_test_pred, "test")
    ]
    save_metrics(metrics)


if __name__ == "__main__":
    main()
