
## Cómo hacer predicciones

### Localmente
```bash
python predict.py                # Usa ejemplo manual
python predict.py nuevo.csv      # Usa datos desde un CSV
```

El resultado se guarda en `predictions.csv`.

### En GitHub Actions
El workflow ahora también ejecuta `predict.py` después del entrenamiento y sube `predictions.csv` como artefacto.
