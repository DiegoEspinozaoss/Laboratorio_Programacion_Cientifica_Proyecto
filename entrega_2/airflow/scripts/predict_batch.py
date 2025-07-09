import pandas as pd
import joblib
from pathlib import Path
import re

print("Iniciando predicción para el batch...")

data_dir = Path("data")
batch_files = list(data_dir.glob("batch_t*.parquet"))

model = joblib.load("models/decision_tree_model.joblib")

if not batch_files:
    print("No se encontraron archivos batch_tN.parquet. Realizando predicción sobre semana futura desde el histórico.")

    transacciones = pd.read_parquet("data/transacciones.parquet")
    transacciones['purchase_date'] = pd.to_datetime(transacciones['purchase_date'])
    transacciones = transacciones[transacciones['items'] > 0]
    transacciones['semana'] = transacciones['purchase_date'].dt.to_period("W").dt.start_time

    clientes = pd.read_parquet("data/clientes.parquet")
    productos = pd.read_parquet("data/productos.parquet")

    clientes_unicos = transacciones['customer_id'].unique()
    productos_unicos = transacciones['product_id'].unique()
    semanas_unicas = transacciones['semana'].unique()

    full_grid = (
        pd.MultiIndex.from_product(
            [clientes_unicos, productos_unicos, semanas_unicas],
            names=["customer_id", "product_id", "semana"]
        ).to_frame(index=False)
    )

    full_grid = full_grid.merge(clientes, on='customer_id', how='left')
    full_grid = full_grid.merge(productos, on='product_id', how='left')
    full_grid['purchase_date'] = pd.to_datetime(full_grid['semana'])

    ultima_semana = full_grid['semana'].max()
    semana_prediccion = pd.to_datetime(ultima_semana) + pd.Timedelta(weeks=1)

    df_semana_prediccion = full_grid[full_grid['semana'] == semana_prediccion].copy()

    if df_semana_prediccion.empty:
        print("⚠️ No hay registros en la grilla para la semana futura. Generando CSV vacío.")
        pares_predichos = pd.DataFrame(columns=['customer_id', 'product_id'])
    else:
        print(f"Registros en la semana a predecir: {len(df_semana_prediccion)}")
        y_pred = model.predict(df_semana_prediccion)
        df_semana_prediccion['predicted_target'] = y_pred
        pares_predichos = df_semana_prediccion[df_semana_prediccion['predicted_target'] == 1][['customer_id', 'product_id']].drop_duplicates()

    output_dir = Path("/opt/airflow/data/pares_generados")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "pares_pred_batch_t0.csv"
    pares_predichos.to_csv(output_path, index=False, header=False)
    print(f"CSV generado: {output_path}")

else:
    def extract_batch_number(filename):
        match = re.search(r'batch_t(\d+)\.parquet', filename.name)
        return int(match.group(1)) if match else -1

    batch_files_sorted = sorted(batch_files, key=extract_batch_number, reverse=True)
    selected_batch = batch_files_sorted[0]
    batch_name = selected_batch.stem

    print(f"Batch seleccionado automáticamente: {selected_batch.name}")

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
