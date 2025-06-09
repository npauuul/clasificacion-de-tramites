from flask import Flask, request, render_template
import joblib
import PyPDF2
import pandas as pd
import unicodedata
from flask import redirect, url_for

app = Flask(__name__)

# Cargar modelo y codificadores
model = joblib.load('models/modelo_multisalida.pkl')
label_encoder = joblib.load('models/label_encoder_tipo_tramite.pkl')
prioridad_map = joblib.load('models/prioridad_map.pkl')
inv_prioridad_map = {v: k for k, v in prioridad_map.items()}


def data(tramite_nombre):
    desconocido = False
    # Quita el prefijo "Asunto:" si está presente
    if tramite_nombre.lower().startswith("asunto:"):
        tramite_nombre = tramite_nombre[len("asunto:"):].strip()
    # Convierte el nombre del trámite a minúsculas, elimina tildes y reemplaza espacios por guiones bajos
    tramite_nombre_normalizado = tramite_nombre.lower()
    tramite_nombre_normalizado = ''.join(
        c for c in unicodedata.normalize('NFD', tramite_nombre_normalizado)
        if unicodedata.category(c) != 'Mn'
    )
    tramite_nombre_normalizado = tramite_nombre_normalizado.replace(" ", "_")
    
    try:
        # Inferencia simple del tipo
        tramite_encoded = label_encoder.transform([tramite_nombre_normalizado]).reshape(-1, 1)
        # Predicción
        pred = model.predict(tramite_encoded)[0]
        prioridad_pred = inv_prioridad_map.get(round(pred[0]), "media")
        tiempo_resolucion_pred = round(pred[1])
    except Exception:
        # Si el trámite no es reconocido, valores por defecto
        prioridad_pred = "media"
        tiempo_resolucion_pred = 5
        desconocido = True

    return prioridad_pred, tramite_nombre, tiempo_resolucion_pred, desconocido

tramites_memoria = []  # Lista global para almacenar los trámites temporalmente

@app.route('/', methods=['GET', 'POST'])
def index():
    prioridad = None
    tramite = None
    tiempo_estimado = None
    error = None
    desconocido = False

    if request.method == 'POST':
        if 'pdf_file' not in request.files:
            error = 'No se subió ningún archivo'
        else:
            file = request.files['pdf_file']
            if file and file.filename.endswith('.pdf'):
                try:
                    pdf_reader = PyPDF2.PdfReader(file)
                    first_page = pdf_reader.pages[0]
                    text = first_page.extract_text()
                    if not text:
                        error = "No se pudo extraer texto del PDF."
                    else:
                        asunto_line = text.splitlines()[0]
                        prioridad, tramite, tiempo_estimado, desconocido = data(asunto_line)
                        # Guardar en memoria
                        tramites_memoria.append({
                            "tramite": tramite,
                            "dias_resolucion": tiempo_estimado,
                            "prioridad": prioridad,
                            "nombres": request.form.get("nombres", ""),
                            "telefono": request.form.get("telefono", "")
                        })
                except Exception as e:
                    error = f"Error leyendo el PDF: {e}"
            else:
                error = 'El archivo no es un PDF válido'
    return render_template(
        'index.html',
        prioridad=prioridad,
        tramite=tramite,
        tiempo_estimado=tiempo_estimado,
        error=error,
        desconocido=desconocido,
        tramites=tramites_memoria  # Pasar la lista a la plantilla
    )

@app.route('/editar', methods=['POST'])
def editar():
    idx = int(request.form.get('idx', -1))
    if 0 <= idx < len(tramites_memoria):
        if 'nombres' in request.form:
            tramites_memoria[idx]['nombres'] = request.form['nombres']
        if 'telefono' in request.form:
            tramites_memoria[idx]['telefono'] = request.form['telefono']
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=False)