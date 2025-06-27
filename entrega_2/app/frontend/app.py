import gradio as gr
import requests

BACKEND = "http://backend:8000"

def load_options():
    try:
        customers = requests.get(f"{BACKEND}/customers").json()["customers"]
        products  = requests.get(f"{BACKEND}/products").json()["products"]
        week_info = requests.get(f"{BACKEND}/current_week").json()
        semana_str = f"🗓️ Semana del {week_info['semana_inicio']} al {week_info['semana_fin']}"
    except Exception as e:
        print("Error cargando opciones:", e)
        customers, products, semana_str = [], [], ""
    return (
        gr.update(choices=customers, value=customers[0] if customers else None),
        gr.update(choices=products,  value=products[0]  if products  else None),
        semana_str
    )

def predict_one(customer_id, product_id):
    payload = {"customer_id": str(customer_id), "product_id": str(product_id)}
    try:
        # 1. Predicción individual
        r1 = requests.post(f"{BACKEND}/predict_single", json=payload)
        r1.raise_for_status()
        data = r1.json()

        pred = data["prediction"]
        pred_msg = "✅ Comprará" if pred == 1 else "❌ No comprará"
        resultado = f"{pred_msg} el producto {product_id}"

        # 2. Otros compradores del producto
        r2 = requests.get(f"{BACKEND}/buyers_for_product/{product_id}")
        r2.raise_for_status()
        buyers = r2.json().get("buyers", [])
        other_buyers = [cid for cid in buyers if str(cid) != str(customer_id)]

        if buyers:
            if pred == 1:
                if other_buyers:
                    compradores = "\n".join(f"👤 {cid}" for cid in other_buyers)
                    lista_compras = f"🧾 Otros clientes que también comprarán el producto {product_id}:\n{compradores}"
                else:
                    lista_compras = "🧾 Ningún otro cliente más lo comprará esta semana."
            else:
                compradores = "\n".join(f"👤 {cid}" for cid in buyers)
                lista_compras = f"🧾 Clientes que sí comprarán el producto {product_id}:\n{compradores}"
        else:
            lista_compras = f"⚠️ Nadie comprará el producto {product_id} esta semana."

        extra = lista_compras
        return resultado, extra

    except requests.exceptions.HTTPError as e:
        msg = f"❌ Error {r1.status_code if 'r1' in locals() else '?'}: {e.response.text}"
        return msg, ""
    except Exception as e:
        return f"❌ Error de conexión: {str(e)}", ""


with gr.Blocks(title="SodAI Drinks 🥤") as demo:
    gr.Markdown("### SodAI Drinks 🥤")
    semana_text = gr.Markdown("")  # aquí irá la semana

    gr.Markdown("Selecciona cliente y producto y predice compra la próxima semana.")

    with gr.Row():
        customer = gr.Dropdown(label="Cliente ID", choices=[], allow_custom_value=False)
        product  = gr.Dropdown(label="Producto ID", choices=[], allow_custom_value=False)

    output = gr.Textbox(label="Resultado")
    extra  = gr.Markdown("")  # debajo de la caja

    btn = gr.Button("Predecir")

    btn.click(fn=predict_one, inputs=[customer, product], outputs=[output, extra])

    demo.load(fn=load_options, inputs=[], outputs=[customer, product, semana_text])

demo.launch(server_name="0.0.0.0", server_port=7860)