import numpy as np
from sistema.sistema import numero_positivos
import os

orden_presentacion = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11"]
orden_presentacion_objetos = ['o1', 'o10', 'o11', 'o12', 'o13', 'o14', 'o15', 'o16', 'o17', 'o18', 'o19', 'o2', 'o20', 'o21', 'o22', 'o23', 'o24', 'o25', 'o26', 'o27', 'o28', 'o29', 'o3', 'o30', 'o31', 'o32', 'o33', 'o34', 'o35', 'o36', 'o37', 'o38', 'o39', 'o4', 'o40', 'o41', 'o42', 'o43', 'o44', 'o45', 'o46', 'o47', 'o48', 'o49', 'o50', 'o5', 'o6', 'o7', 'o8', 'o9']

def cargar_CORe50(directorio: str) -> list:
    datos = []
    print("[", end="", flush=True)
    for nombre_escena in orden_presentacion:
        print("-", end="", flush=True)
        escena = []
        for objeto in orden_presentacion_objetos: # o1, o2...
            secuencia = []
            for imagen_objeto in os.listdir(directorio + '/' + nombre_escena + '/' + objeto): # 1.jpg, 2.jpg...
                secuencia.append(np.fromfile(directorio + '/' + nombre_escena + '/' + objeto + '/' + imagen_objeto, dtype=np.float32))
            escena.append(secuencia)

        inicio = 0
        if nombre_escena == orden_presentacion[0]:
            escena_homogenea = []
            for secuencia in escena:
                escena_homogenea.append(np.array(secuencia[:numero_positivos], dtype = np.float32))
            datos.append(escena_homogenea)
            inicio = numero_positivos

        longitud_minima = 300
        for secuencia in escena:
            if len(secuencia[inicio:]) < longitud_minima:
                longitud_minima = len(secuencia)
        escena_homogenea = []
        for secuencia in escena: 
            escena_homogenea.append(np.array(secuencia[inicio:longitud_minima], dtype = np.float32))
        datos.append(escena_homogenea)
    print("]")
    return datos