import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

# Cargar datos
data = pd.read_csv('data.csv')

# Preprocesamiento
# Convertir fecha y eliminar filas nulas
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')
data = data.dropna(subset=['maxtemp', 'mintemp', 'pressure', 'humidity', 'mean wind speed'])

# Seleccionar variables
X = data[['mintemp', 'pressure', 'humidity', 'mean wind speed']]
y = data['maxtemp']

# Dividir en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
model = RandomForestRegressor(n_estimators=500, random_state=42)
model.fit(X_train, y_train)

# Evaluación
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
rmse = mean_squared_error(y_test, preds) ** 0.5
r2 = r2_score(y_test, preds)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# Guardar modelo
joblib.dump(model, 'model.pkl')

# Importancia de variables
importances = model.feature_importances_
features = X.columns
plt.figure(figsize=(8,6))
plt.barh(features, importances)
plt.xlabel('Importancia')
plt.title('Importancia de Variables en Random Forest')
plt.tight_layout()
plt.savefig('feature_importance.png')
