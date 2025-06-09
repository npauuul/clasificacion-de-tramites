import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib

# Cargar datos
df = pd.read_csv('tramites_municipales.csv')

# Codificar el input
label_encoder = LabelEncoder()
df['tipo_tramite_encoded'] = label_encoder.fit_transform(df['tipo_tramite'])

# Codificar la prioridad como número para regresión multisalida
prioridad_map = {'muy_alta': 3, 'alta': 2, 'media': 1, 'baja': 0}
df['prioridad_numerica'] = df['prioridad'].map(prioridad_map)

# Targets: prioridad (numérica) y tiempo de resolución
y = df[['prioridad_numerica', 'tiempo_resolucion_dias']]
X = df[['tipo_tramite_encoded']]

# Modelo multisalida (RandomForestRegressor soporta MultiOutputRegressor)
model = MultiOutputRegressor(RandomForestRegressor())
model.fit(X, y)

# Guardar modelo, codificador y mapeo de prioridad
joblib.dump(model, 'models/modelo_multisalida.pkl')
joblib.dump(label_encoder, 'models/label_encoder_tipo_tramite.pkl')
joblib.dump(prioridad_map, 'models/prioridad_map.pkl')