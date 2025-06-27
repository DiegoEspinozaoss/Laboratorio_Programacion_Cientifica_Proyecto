from fastapi import FastAPI
from datetime import datetime, timedelta
from fastapi.responses import JSONResponse
import joblib
import pandas as pd
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils import DateFeatureExtractor  

app = FastAPI(title="SodAI Drinks 🥤 - Backend")

# Carga
model        = joblib.load("models/decision_tree_model.joblib")
clientes_df  = pd.read_parquet("data/clientes.parquet").set_index("customer_id")
productos_df = pd.read_parquet("data/productos.parquet").set_index("product_id")

def format_product(row):
    return f"{row['brand']} | {row['sub_category']} {row['segment']} | {row['package']} {row['size']:.2f}L"

productos_df["product_name"] = productos_df.apply(format_product, axis=1)

@app.get("/")
def read_root():
    return {"message": "SodAI Drinks Backend está corriendo 🚀"}

@app.get("/customers")
def list_customers():
    return {"customers": clientes_df.index.unique().tolist()}

@app.get("/products")
def list_products():
    return {"products": productos_df.index.unique().tolist()}

@app.get("/current_week")
def current_week():
    next_week = pd.Timestamp.now().normalize() + pd.Timedelta(days=7)
    start = next_week.to_period("W").start_time
    end   = start + timedelta(days=6)
    return JSONResponse({"semana_inicio": str(start.date()), "semana_fin": str(end.date())})

@app.get("/recommend_products")
def recommend_products(customer_id: str, top_k: int = 5):
    next_week = pd.Timestamp.now().normalize() + pd.Timedelta(days=7)
    semana = next_week.to_period("W").start_time

    df = pd.DataFrame({
        "customer_id":   [customer_id] * len(productos_df),
        "product_id":    productos_df.index,
        "semana":        semana,
        "purchase_date": next_week
    })

    df = (
        df.set_index(["customer_id", "product_id"])
          .join(clientes_df, how="left")
          .join(productos_df, how="left")
          .reset_index()
    )

    df["score"] = model.predict(df)

    recomendados = (
        df[df["score"] == 1]
        .head(top_k)
        .loc[:, ["product_id", "product_name"]]
    )

    return {
        "recommended": [
            {
                "product_id": str(row["product_id"]),
                "product_name": row["product_name"]
            }
            for _, row in recomendados.iterrows()
        ]
    }
