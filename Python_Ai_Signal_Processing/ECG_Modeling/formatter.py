import pandas as pd
import sys

def dividir_archivo_emg(nombre_archivo):
    try:
        # Leer el archivo CSV
        datos = pd.read_csv(nombre_archivo)
        
        # Verificar que el archivo tiene las columnas esperadas
        columnas_esperadas = ['timestamp', 'emg1', 'emg2', 'emg3', 'emg4', 'emg5', 'emg6', 'emg7', 'emg8']
        if not all(col in datos.columns for col in columnas_esperadas):
            print(f"El archivo {nombre_archivo} no contiene las columnas esperadas.")
            return
        
        # Iterar sobre las columnas EMG y guardar cada una como archivo individual
        for emg_col in columnas_esperadas[1:]:  # Saltar 'timestamp'
            subdatos = datos[['timestamp', emg_col]].rename(columns={'timestamp': 'Time', emg_col: 'Signal'})
            nombre_salida = f"{emg_col}.csv"
            subdatos.to_csv(nombre_salida, index=False)
            print(f"Archivo {nombre_salida} generado con éxito.")
    
    except Exception as e:
        print(f"Error procesando el archivo {nombre_archivo}: {e}")

# Uso del script
if __name__ == "__main__":
    # Cambia "Test_Data.csv" por el nombre del archivo si es diferente
    archivo_entrada = "Test_Data.csv"
    dividir_archivo_emg(archivo_entrada)
