import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils import DateFeatureExtractor

import pandas as pd
import joblib
import mlflow
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from datetime import datetime, timedelta
from pathlib import Path
import re
import gc

transacciones = pd.read_parquet("data/transacciones.parquet")
transacciones['purchase_date'] = pd.to_datetime(transacciones['purchase_date'])

data_dir = Path("data")
batch_files = list(data_dir.glob("batch_t*.parquet"))

for batch_file in sorted(batch_files, key=lambda x: int(re.search(r'batch_t(\d+)\.parquet', x.name).group(1))):
    print(f"✅ Añadiendo datos del batch: {batch_file.name}")
    batch = pd.read_parquet(batch_file)
    batch['purchase_date'] = pd.to_datetime(batch['purchase_date'], unit='ms')
    transacciones = pd.concat([transacciones, batch], ignore_index=True)

print("✅ Todos los datos combinados correctamente. Total registros:", len(transacciones))

clientes = pd.read_parquet("data/clientes.parquet")
productos = pd.read_parquet("data/productos.parquet")

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

print("📊 Dataset total:", len(df))


del transacciones, full_grid, compras_realizadas
gc.collect()
print("🩹 Memoria liberada (transacciones + full_grid eliminados).")


min_date = df['purchase_date'].min()
max_date = df['purchase_date'].max()
total_days = (max_date - min_date).days

cutoff_train = min_date + timedelta(days=int(total_days * 0.6))
cutoff_valid = min_date + timedelta(days=int(total_days * 0.8))

df_train = df[df['purchase_date'] <= cutoff_train]
df_valid = df[(df['purchase_date'] > cutoff_train) & (df['purchase_date'] <= cutoff_valid)]

print("✔️ Tamaño entrenamiento:", len(df_train))
print("✔️ Tamaño validación:", len(df_valid))

MAX_TRAIN_RATIO = 1
MAX_VALID_RATIO = 1

max_train_rows = int(len(df_train) * MAX_TRAIN_RATIO)
max_valid_rows = int(len(df_valid) * MAX_VALID_RATIO)

if len(df_train) > max_train_rows:
    print(f"⚠️ Limite aplicado: entrenamiento reducido al {MAX_TRAIN_RATIO*100:.1f}% ({max_train_rows:,} registros)")
    df_train = df_train.sample(n=max_train_rows, random_state=42)

if len(df_valid) > max_valid_rows:
    print(f"⚠️ Limite aplicado: validación reducida al {MAX_VALID_RATIO*100:.1f}% ({max_valid_rows:,} registros)")
    df_valid = df_valid.sample(n=max_valid_rows, random_state=42)

categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
excluded_cols = ['target', 'items', 'purchase_date']
categorical_cols = [col for col in categorical_cols if col not in excluded_cols]
numerical_cols = [col for col in numerical_cols if col not in excluded_cols]

X_train = df_train.drop(columns=['target'])
y_train = df_train['target']
X_valid = df_valid.drop(columns=['target'])
y_valid = df_valid['target']

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

print("🚀 Generando predicción SOLO para la semana inmediatamente posterior...")

ultima_semana_entrenada = df['semana'].max()
semana_prediccion = pd.to_datetime(ultima_semana_entrenada) + pd.Timedelta(weeks=1)

print(f"➞ Semana de predicción: {semana_prediccion.date()}")

df_semana_prediccion = df[df['semana'] == semana_prediccion].copy()

if df_semana_prediccion.empty:
    print("No hay registros en la grilla para la semana a predecir. Revisa tus datos.")
else:
    print(f"Registros en la semana a predecir: {len(df_semana_prediccion)}")

    y_pred = pipeline.predict(df_semana_prediccion.drop(columns=['target']))
    df_semana_prediccion['predicted_target'] = y_pred

    pares_predichos = df_semana_prediccion[df_semana_prediccion['predicted_target'] == 1][['customer_id', 'product_id']].drop_duplicates()

    output_dir = Path("/opt/airflow/data/pares_generados")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"prediccion_semana_{semana_prediccion.strftime('%Y-%m-%d')}.csv"

    pares_predichos.to_csv(output_path, index=False, header=False)
    print(f"CSV generado para entregar: {output_path}")
