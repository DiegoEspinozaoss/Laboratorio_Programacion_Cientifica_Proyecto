import gradio as gr
import requests

BACKEND = "http://backend:8000"

def load_options():
    try:
        customers = requests.get(f"{BACKEND}/customers").json()["customers"]
        week_info = requests.get(f"{BACKEND}/current_week").json()
        semana_str = f"🗓️ Semana del {week_info['semana_inicio']} al {week_info['semana_fin']}"
    except Exception as e:
        print("Error cargando opciones:", e)
        customers, semana_str = [], ""
    return (
        gr.update(choices=customers, value=customers[0] if customers else None),
        semana_str
    )

def recommend_only(customer_id):
    try:
        r = requests.get(f"{BACKEND}/recommend_products", params={"customer_id": customer_id})
        r.raise_for_status()
        recomendaciones = r.json().get("recommended", [])

        if recomendaciones:
            lista_recom = "\n".join(
                f"🥤 {p['product_id']} — {p['product_name']}" for p in recomendaciones
            )
            return f"🧠 Recomendaciones para cliente {customer_id}:\n{lista_recom}"
        else:
            return f"⚠️ No hay productos recomendados para {customer_id} esta semana."
    except Exception as e:
        return f"❌ Error al obtener recomendaciones: {str(e)}"

with gr.Blocks(title="SodAI Drinks 🥤") as demo:
    gr.Markdown("### SodAI Drinks 🥤")
    semana_text = gr.Markdown("")

    gr.Markdown("Selecciona cliente para recibir recomendaciones.")

    customer = gr.Dropdown(label="Cliente ID", choices=[], allow_custom_value=False)
    output   = gr.Textbox(label="Resultado")
    btn      = gr.Button("Recomendar")

    btn.click(fn=recommend_only, inputs=[customer], outputs=[output])
    demo.load(fn=load_options, inputs=[], outputs=[customer, semana_text])

demo.launch(server_name="0.0.0.0", server_port=7860)
