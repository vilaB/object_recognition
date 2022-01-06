import numpy as np
import os


def cargar_CORe50(directorio: str) -> list:
    datos = []
    for escena in os.listdir(directorio): # S1, S2...
        escena = {'escena': escena, 'objetos': []}
        for objeto in os.listdir(directorio + '/' + escena): # o1, o2...
            secuencia = {'objeto': objeto, 'imagenes': []}
            for imagen_objeto in os.listdir(directorio + '/' + escena + '/' + objeto): # 1.jpg, 2.jpg...
                secuencia['imagenes'].append(np.fromfile(directorio + '/' + escena + '/' + objeto + '/' + imagen_objeto, dtype=np.float32))
            secuencia['imagenes'] = np.array(secuencia['imagenes'], dtype = np.float32)
            escena['objetos'].append(secuencia)
        datos.append(escena)
        
    return datos