import pandas as pd
import joblib
from pathlib import Path
import re

print("Iniciando pred. para el batch...")

data_dir = Path("data")
batch_files = list(data_dir.glob("batch_t*.parquet"))

if not batch_files:
    raise FileNotFoundError("No se encontraron archivos batch_tN.parquet en data/")

def extract_batch_number(filename):
    match = re.search(r'batch_t(\d+)\.parquet', filename.name)
    return int(match.group(1)) if match else -1

batch_files_sorted = sorted(batch_files, key=extract_batch_number, reverse=True)
selected_batch = batch_files_sorted[0]
batch_name = selected_batch.stem

print(f"Batch seleccionado automáticamente: {selected_batch.name}")

model = joblib.load("models/decision_tree_model.joblib")

batch = pd.read_parquet(selected_batch)
batch['purchase_date'] = pd.to_datetime(batch['purchase_date'], unit='ms')
batch['semana'] = batch['purchase_date'].dt.to_period("W").dt.start_time

clientes = pd.read_parquet("data/clientes.parquet")
productos = pd.read_parquet("data/productos.parquet")
batch = batch.merge(clientes, on='customer_id', how='left')
batch = batch.merge(productos, on='product_id', how='left')

X_batch = batch.drop(columns=['order_id', 'items'])
y_pred = model.predict(X_batch)

batch['predicted_target'] = y_pred

output_dir = Path("/opt/airflow/data/pares_generados")
output_dir.mkdir(exist_ok=True)
pares = batch[batch['predicted_target'] == 1][['customer_id', 'product_id']].drop_duplicates()

output_path = output_dir / f"pares_pred_{batch_name}.csv"
pares.to_csv(output_path, index=False, header=False)
print(f"CSV generado: {output_path}")
