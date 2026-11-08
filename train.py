import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

numeric_columns = ["age", "bmi", "children"]
categorical_columns = ["sex", "smoker", "region"]
features = numeric_columns + categorical_columns

data = pd.read_csv(BASE_DIR / "data" / "medical_insurance.csv")
original_rows = len(data)
data = data.drop_duplicates().dropna(subset=["charges"])

X = data[features]
y = data["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns),
])

model_parameters = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": 1,
}

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(**model_parameters)),
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

metrics = {
    "mae": float(mean_absolute_error(y_test, predictions)),
    "rmse": float(root_mean_squared_error(y_test, predictions)),
    "r2": float(r2_score(y_test, predictions)),
}

model_path = ARTIFACTS_DIR / "pipeline.joblib"
joblib.dump(pipeline, model_path, compress=3)

loaded_pipeline = joblib.load(model_path)
sample = X_test.head(2)
np.testing.assert_allclose(
    loaded_pipeline.predict(sample),
    pipeline.predict(sample),
)
print("Guardado y recarga del pipeline: OK")

metadata = {
    "original_rows": original_rows,
    "usable_rows": len(data),
    "train_rows": len(X_train),
    "test_rows": len(X_test),
    "features": features,
    "target": "charges",
    "sklearn_version": sklearn.__version__,
    "model_parameters": model_parameters,
    "metrics": metrics,
}
metadata_path = ARTIFACTS_DIR / "training_metadata.json"
metadata_path.write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8",
)

print("\nPipeline completo:")
print(pipeline)
print("\nResultados:")
print(json.dumps(metadata, indent=2))
print("\nPipeline guardado en:", model_path)
print("Tamano del archivo:", model_path.stat().st_size, "bytes")
