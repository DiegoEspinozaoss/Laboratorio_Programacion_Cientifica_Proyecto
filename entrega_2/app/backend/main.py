from fastapi import FastAPI, Request, Query
from pydantic import BaseModel
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
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

class PredictionInput(BaseModel):
    customer_id: str
    product_id: str

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )

@app.get("/")
def read_root():
    return {"message": "SodAI Drinks Backend está corriendo 🚀"}

@app.get("/customers")
def list_customers():
    return {"customers": clientes_df.index.unique().tolist()}

@app.get("/products")
def list_products():
    return {"products": productos_df.index.unique().tolist()}

@app.post("/predict_single")
def predict_single(input_data: PredictionInput):
    next_week = pd.Timestamp.now().normalize() + pd.Timedelta(days=7)
    semana    = next_week.to_period("W").start_time
    semana_fin = semana + pd.Timedelta(days=6)

    df = pd.DataFrame([{
        "customer_id":   str(input_data.customer_id),
        "product_id":    str(input_data.product_id),
        "semana":        semana,
        "purchase_date": next_week
    }])

    df = (
        df.set_index(["customer_id", "product_id"])
          .join(clientes_df,  how="left")
          .join(productos_df, how="left")
          .reset_index()
    )

    pred = model.predict(df)[0]
    return {
        "prediction": int(pred),
        "semana_inicio": semana.strftime("%Y-%m-%d"),
        "semana_fin": semana_fin.strftime("%Y-%m-%d")
    }

@app.get("/buyers_for_product/{product_id}")
def buyers_for_product(product_id: str):
    next_week = pd.Timestamp.now().normalize() + pd.Timedelta(days=7)
    semana = next_week.to_period("W").start_time

    customers = clientes_df.index.unique()
    df = pd.DataFrame([{
        "customer_id": str(cust),
        "product_id": str(product_id),
        "semana": semana,
        "purchase_date": next_week
    } for cust in customers])

    df = (
        df.set_index(["customer_id", "product_id"])
          .join(clientes_df,  how="left")
          .join(productos_df, how="left")
          .reset_index()
    )

    preds = model.predict(df)
    compradores = df.loc[preds == 1, "customer_id"].tolist()

    return {
        "product_id": product_id,
        "buyers": compradores,
        "semana_inicio": semana.strftime("%Y-%m-%d"),
        "semana_fin": (semana + pd.Timedelta(days=6)).strftime("%Y-%m-%d")
    }

@app.get("/current_week")
def current_week():
    next_week = pd.Timestamp.now().normalize() + pd.Timedelta(days=7)
    start = next_week.to_period("W").start_time
    end   = start + timedelta(days=6)
    return JSONResponse({"semana_inicio": str(start.date()), "semana_fin": str(end.date())})
