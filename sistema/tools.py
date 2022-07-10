import numpy as np
from sistema.constantes import FUNCION_FDR, FUNCION_SDR, FUNCION_DECISION_COMITE_GANADOR
import statistics

numero_positivos=10
numero_negativos=100
umbral_reconocimiento=np.inf            # np.inf para closed set!!
funcion_FDR = FUNCION_FDR.PERCENTIL       # Función a nivel de comité
percentil_FDR = 0.16            
modo_SDR = FUNCION_SDR.MEDIANA           # Función a nivel de secuencia
percentil_SDR = 0.25
tamano_maximo_comite = 18
funcion_decision_comite_ganador = FUNCION_DECISION_COMITE_GANADOR.EL_MEJOR   # Mayor respuesta o weibull

def generar_negativos(muestras_inicializacion: list, posicion_positivo: int):
    negativos = np.array(muestras_inicializacion[0:posicion_positivo] + muestras_inicializacion[posicion_positivo + 1:]) # Cogemos como negativos todas las demás secuencias menos la propa: usamos esta aritmética de listas para evitar hacer una deepcopy
    negativos = np.vstack(negativos[:])
    negativos = np.vstack([negativos])
    return negativos


# Función a nivel de comité (obtener una puntuación por imagen) 
# In: miembros_comite x num_imágenes
# Out: num_imágenes
def FDR(puntuaciones_de_un_comite):
    if funcion_FDR == FUNCION_FDR.MEDIANA:     puntuaciones_imagenes = np.median(puntuaciones_de_un_comite, axis=0)
    elif funcion_FDR == FUNCION_FDR.PERCENTIL: puntuaciones_imagenes = np.quantile(puntuaciones_de_un_comite, percentil_FDR, axis=0)
    elif funcion_FDR == FUNCION_FDR.EL_MEJOR:  puntuaciones_imagenes = np.min(puntuaciones_de_un_comite, axis=0)
    elif funcion_FDR == FUNCION_FDR.FIJO:
        if puntuaciones_de_un_comite.shape[0] >= 3:
            puntuaciones_imagenes = np.sort(puntuaciones_de_un_comite, axis=0)[2, :]
        elif puntuaciones_de_un_comite.shape[0] == 2:
            puntuaciones_imagenes = np.sort(puntuaciones_de_un_comite, axis=0)[1, :]
        else:
            puntuaciones_imagenes = np.sort(puntuaciones_de_un_comite, axis=0)[0, :]
    return puntuaciones_imagenes

# Función a nivel de secuencia (obtener una puntuación por comité)
# In: num_imágenes
# Out: 1
def SDR(puntuaciones_imagenes):
    if modo_SDR == FUNCION_SDR.MEDIANA:      puntuacion_comite = statistics.median(puntuaciones_imagenes)
    elif modo_SDR == FUNCION_SDR.PERCENTIL:  puntuacion_comite = np.quantile(puntuaciones_imagenes, percentil_SDR)
    return puntuacion_comite