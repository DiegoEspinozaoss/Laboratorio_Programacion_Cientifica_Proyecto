import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils import DateFeatureExtractor  # ✅ usar versión oficial

import pandas as pd
import joblib
import mlflow
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from datetime import datetime, timedelta

# -------------------- PREPROCESAMIENTO --------------------

clientes = pd.read_parquet("data/clientes.parquet")
productos = pd.read_parquet("data/productos.parquet")
transacciones = pd.read_parquet("data/transacciones.parquet")

transacciones['purchase_date'] = pd.to_datetime(transacciones['purchase_date'])
transacciones = transacciones[transacciones['items'] > 0]
transacciones['semana'] = transacciones['purchase_date'].dt.to_period("W").dt.start_time

clientes_unicos = transacciones['customer_id'].unique()
productos_unicos = transacciones['product_id'].unique()
semanas_unicas = transacciones['semana'].unique()

full_grid = (
    pd.MultiIndex.from_product(
        [clientes_unicos, productos_unicos, semanas_unicas],
        names=["customer_id", "product_id", "semana"]
    ).to_frame(index=False)
)

compras_realizadas = (
    transacciones.groupby(['customer_id', 'product_id', 'semana'])['items']
    .sum().reset_index()
)
compras_realizadas['target'] = 1

df = full_grid.merge(compras_realizadas[['customer_id', 'product_id', 'semana', 'target']],
                     on=['customer_id', 'product_id', 'semana'],
                     how='left')
df['target'] = df['target'].fillna(0).astype(int)

df = df.merge(clientes, on='customer_id', how='left')
df = df.merge(productos, on='product_id', how='left')
df['purchase_date'] = pd.to_datetime(df['semana'])

# -------------------- HOLDOUT SPLIT --------------------

min_date = df['purchase_date'].min()
max_date = df['purchase_date'].max()
total_days = (max_date - min_date).days

cutoff_train = min_date + timedelta(days=int(total_days * 0.6))
cutoff_valid = min_date + timedelta(days=int(total_days * 0.8))

df_train = df[df['purchase_date'] <= cutoff_train]
df_valid = df[(df['purchase_date'] > cutoff_train) & (df['purchase_date'] <= cutoff_valid)]
df_test  = df[df['purchase_date'] > cutoff_valid]

X_train = df_train.drop(columns=['target'])
X_train = X_train.sample(n=min(3_000_000, len(X_train)))
y_train = df_train['target']
y_train = y_train.loc[X_train.index]

X_valid = df_valid.drop(columns=['target'])
y_valid = df_valid['target']
X_test  = df_test.drop(columns=['target'])
y_test  = df_test['target']

# -------------------- FEATURE ENGINEERING --------------------

categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
excluded_cols = ['target', 'items', 'purchase_date']
categorical_cols = [col for col in categorical_cols if col not in excluded_cols]
numerical_cols = [col for col in numerical_cols if col not in excluded_cols]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numerical_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# -------------------- ENTRENAMIENTO --------------------

DT = DecisionTreeClassifier(
    criterion='gini',
    splitter='best',
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight='balanced',
)

pipeline = Pipeline([
    ("date_features", DateFeatureExtractor(date_column="purchase_date")),
    ("preprocessor", preprocessor),
    ("clf", DT)
])

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("DecisionTree Pipeline")

with mlflow.start_run(run_name=f"tree_run_{datetime.today().strftime('%Y%m%d')}"):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_valid)
    
    f1 = f1_score(y_valid, y_pred, average='weighted')
    mlflow.log_metric("f1_score_val", f1)
    mlflow.sklearn.log_model(pipeline, "model")
    mlflow.log_params(DT.get_params())

    print("Reporte de clasificación (validación):")
    print(classification_report(y_valid, y_pred))

    joblib.dump(pipeline, "models/decision_tree_model.joblib")
