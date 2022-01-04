from comite import Comite
import numpy as np
from comite import construir_muestra_de_entrenamiento
import statistics


numero_positivos=25
numero_negativos=100
umbral_actualizacion=np.inf
funcion_FDR = 'percentil'       # Función a nivel de comité
percentil_FDR = 0.16            
modo_SDR = 'mediana'            # Función a nivel de secuencia
percentil_SDR = 0.25


# TODO: healing

class Sistema():
    comites_no_supervisados: list[Comite] = []
    comites_supervisados: list[Comite] = []
    muestra_de_inicializacion = None

    def __init__(self, muestra_de_inicializacion: list):
        print("Construcción del sistema en curso...")
        self.muestra_de_inicializacion = muestra_de_inicializacion
        for individuo in range(len(muestra_de_inicializacion)):
            negativos = generar_negativos(muestra_de_inicializacion, individuo)
            comite_no_supervisado = Comite(positivos=muestra_de_inicializacion[individuo], negativos=negativos, numero_positivos=numero_positivos, numero_negativos=numero_negativos)
            comite_supervisado = Comite(positivos=muestra_de_inicializacion[individuo], negativos=negativos, numero_positivos=numero_positivos, numero_negativos=numero_negativos)
            self.comites_no_supervisados.append(comite_no_supervisado)
            self.comites_supervisados.append(comite_supervisado)
        print("Construcción del sistema finalizada.")


    def entrenar(self, secuencia: list, individuo: int):
        prediccion = self.entrenamiento_no_supervisado(secuencia)
        self.comites_no_supervisados[prediccion].purgar_comite()
        self.entrenamiento_supervisado(secuencia, individuo)
        return prediccion

    
    def entrenamiento_no_supervisado(self, secuencia: list):
        # Predicción por parte del sistema no supervisado
        puntuaciones_comites, puntuaciones_imagenes_de_comites = self.__presentar_secuencia(secuencia, self.comites_no_supervisados)
        if min(puntuaciones_comites) < umbral_actualizacion:
            prediccion = np.argmin(puntuaciones_comites)
        else:
            prediccion = -1

        if prediccion >= 0:
            puntuaciones_imagenes = puntuaciones_imagenes_de_comites[prediccion]
            puntuaciones_imagenes = np.array(puntuaciones_imagenes)
            puntuaciones_imagenes = np.array([abs(x) for x in puntuaciones_imagenes])
            indices_ordenados = np.argsort(puntuaciones_imagenes)                       # Nos devuelve una lista con las posiciones con las puntuaciones más bajas (+ cercanas a la frontera del conocimiento)
            positivos = []
            for indice in indices_ordenados[:numero_positivos]:
                positivos.append(secuencia[indice, :].reshape(1, -1))
            indice = indices_ordenados[numero_positivos - 1]
            positivos = np.vstack(positivos)
            negativos = generar_negativos(self.muestra_de_inicializacion, prediccion)
            self.comites_no_supervisados[prediccion].entrenamiento(positivos, negativos, numero_positivos, numero_negativos)
            return prediccion
    

    def entrenamiento_supervisado(self, secuencia: list, individuo: int):
        comite = self.comites_supervisados[individuo]
        matriz_del_comite = comite.procesar_secuencia(secuencia)  # Devolve unha lista coa puntuación que lle da cada un dos ensembles do IoI
        puntuaciones_imagenes = self.__FDR(matriz_del_comite)  # Calcula la puntuación final, por ejemplo, con la media de la lista

        puntuaciones_imagenes = np.array(puntuaciones_imagenes)
        puntuaciones_imagenes = np.array([abs(x) for x in puntuaciones_imagenes])
        indices_ordenados = np.argsort(puntuaciones_imagenes)                       # Nos devuelve una lista con las posiciones con las puntuaciones más bajas (+ cercanas a la frontera del conocimiento)
        positivos = []
        for indice in indices_ordenados[:numero_positivos]:
            positivos.append(secuencia[indice, :].reshape(1, -1))
        indice = indices_ordenados[numero_positivos - 1]
        positivos = np.vstack(positivos)
        negativos = generar_negativos(self.muestra_de_inicializacion, individuo)
        comite.entrenamiento(positivos, negativos, numero_positivos, numero_negativos)

    
    def __presentar_secuencia(self, secuencia, comites: list[Comite]):
        puntuaciones_de_cada_comite = []
        puntuaciones_imagenes_de_comites = []
        for comite in comites:
            matriz_del_comite = comite.procesar_secuencia(secuencia)  # Devolve unha lista coa puntuación que lle da cada un dos ensembles do IoI
            puntuaciones_imagenes = self.__FDR(matriz_del_comite)  # Calcula la puntuación final, por ejemplo, con la media de la lista
            puntuaciones_imagenes_de_comites.append(puntuaciones_imagenes)
            puntuacion_del_comite = self.__SDR(puntuaciones_imagenes)
            puntuaciones_de_cada_comite.append(puntuacion_del_comite)
        return puntuaciones_de_cada_comite, puntuaciones_imagenes_de_comites

    
    # Función a nivel de comité (obtener una puntuación por imagen)
    def __FDR(self, puntuaciones_de_un_comite):
        if funcion_FDR == 'mediana':     puntuaciones_imagenes = np.median(puntuaciones_de_un_comite, axis=0)
        elif funcion_FDR == "percentil": puntuaciones_imagenes = np.quantile(puntuaciones_de_un_comite, percentil_FDR, axis=0)
        elif funcion_FDR == "el mejor":  puntuaciones_imagenes = np.min(puntuaciones_de_un_comite, axis=0)
        return puntuaciones_imagenes
    

    # Función a nivel de secuencia (obtener una puntuación por comité)
    def __SDR(self, puntuaciones_imagenes):
        if modo_SDR == 'mediana':      puntuacion_comite = statistics.median(puntuaciones_imagenes)
        elif modo_SDR == 'percentil':  puntuacion_comite = np.quantile(puntuaciones_imagenes, percentil_SDR)
        return puntuacion_comite




def generar_negativos(muestras_inicializacion: list, posicion_positivo: int):
    negativos = np.array(muestras_inicializacion[0:posicion_positivo] + muestras_inicializacion[posicion_positivo + 1:]) # Cogemos como negativos todas las demás secuencias menos la propa: usamos esta aritmética de listas para evitar hacer una deepcopy
    negativos = np.vstack(negativos[:, 0])
    negativos = np.vstack([negativos])
    return negativos