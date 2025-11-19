import joblib
import pandas as pd
import sys

# Cargar modelo entrenado
model = joblib.load('model.pkl')

# Si se pasa un CSV como argumento, usarlo; si no, usar ejemplo manual
if len(sys.argv) > 1:
    input_file = sys.argv[1]
    data = pd.read_csv(input_file)
else:
    # Ejemplo manual
    data = pd.DataFrame([[15.0, 760.0, 88.0, 2.0]], columns=['mintemp','pressure','humidity','mean wind speed'])

# Realizar predicción
preds = model.predict(data)

# Guardar resultados en predictions.csv
output = pd.DataFrame({'Prediccion_maxtemp': preds})
output.to_csv('predictions.csv', index=False)

print("Predicciones guardadas en predictions.csv")
