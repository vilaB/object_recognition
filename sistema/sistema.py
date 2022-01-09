from sistema.comite import Comite
import numpy as np
import statistics


numero_positivos=25
numero_negativos=100
umbral_actualizacion=np.inf
funcion_FDR = 'percentil'       # Función a nivel de comité
percentil_FDR = 0.16            
modo_SDR = 'mediana'            # Función a nivel de secuencia
percentil_SDR = 0.25
tamano_maximo_comite = 18   

class Sistema():
    comites_no_supervisados: list[Comite] = None
    comites_supervisados: list[Comite] = None 
    muestra_de_inicializacion = None

    def __init__(self, muestra_de_inicializacion: list):
        print("Construcción del sistema en curso...")
        self.muestra_de_inicializacion = muestra_de_inicializacion
        self.comites_no_supervisados = []
        self.comites_supervisados = []
        for individuo in range(len(muestra_de_inicializacion)):
            negativos = generar_negativos(muestra_de_inicializacion, individuo)
            comite_no_supervisado = Comite(positivos=muestra_de_inicializacion[individuo], negativos=negativos, numero_positivos=numero_positivos, numero_negativos=numero_negativos)
            comite_supervisado = Comite(positivos=muestra_de_inicializacion[individuo], negativos=negativos, numero_positivos=numero_positivos, numero_negativos=numero_negativos)
            self.comites_no_supervisados.append(comite_no_supervisado)
            self.comites_supervisados.append(comite_supervisado)
        print("Construcción del sistema finalizada.")
        print("Los parámetros del sistema son: ")
        print("\t- Número de positivos (creación SVM): ", numero_positivos)
        print("\t- Número de negativos (creación SVM): ", numero_negativos)
        print("\t- Umbral de actualización: ", umbral_actualizacion)
        print("\t- Función de FDR: ", funcion_FDR)
        print("\t- Percentil de FDR: ", percentil_FDR)
        print("\t- Función de SDR: ", modo_SDR)
        print("\t- Percentil de SDR: ", percentil_SDR)
        print("\t- Tamaño máximo de comité: ", tamano_maximo_comite)

    
    # str method
    def __str__(self):
        return """
        Sistema:
            - Número de positivos (creación SVM): {}
            - Número de negativos (creación SVM): {}
            - Umbral de actualización: {}
            - Función de FDR: {}
            - Percentil de FDR: {}
            - Función de SDR: {}
            - Percentil de SDR: {}
            - Tamaño máximo de comité: {}
            """.format(numero_positivos, numero_negativos, umbral_actualizacion, funcion_FDR, percentil_FDR, modo_SDR, percentil_SDR, tamano_maximo_comite)


    def test(self, secuencia: list, individuo: int):
        puntuaciones_comites_no_supervisados, _ = self.__presentar_secuencia(secuencia, self.comites_no_supervisados)
        prediccion_no_supervisados = self.__funcion_decision_comite_ganador(puntuaciones_comites_no_supervisados)

        puntuaciones_comites_supervisados, _ = self.__presentar_secuencia(secuencia, self.comites_supervisados)
        prediccion_supervisados = self.__funcion_decision_comite_ganador(puntuaciones_comites_supervisados)
        return prediccion_no_supervisados == individuo, prediccion_supervisados == individuo


    def entrenar(self, secuencia: list, individuo: int):
        prediccion = self.entrenamiento_no_supervisado(secuencia)
        self.comites_no_supervisados[prediccion].purgar_comite(tamano_maximo_comite, self.muestra_de_inicializacion)
        self.entrenamiento_supervisado(secuencia, individuo)
        return prediccion

    
    def entrenamiento_no_supervisado(self, secuencia: list):
        # Predicción por parte del sistema no supervisado
        puntuaciones_comites, puntuaciones_imagenes_de_comites = self.__presentar_secuencia(secuencia, self.comites_no_supervisados)
        prediccion = self.__funcion_decision_comite_ganador(puntuaciones_comites)

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
    

    def healing(self):
        for i, comite in enumerate(self.comites_no_supervisados):
            secuencias_positivas = comite.obtener_positivos()
            for j, secuencia in enumerate(secuencias_positivas):                                    # Número de clasificador, secuencia usada para su creación
                puntuaciones, _ = self.__presentar_secuencia(secuencia, self.comites_no_supervisados)
                prediccion = self.__funcion_decision_comite_ganador(puntuaciones)
                if prediccion != i:
                    comite.marcar_miembro_para_eliminar(j)
        for comite in self.comites_no_supervisados:
            comite.eliminar_miembros_marcados()


    def __funcion_decision_comite_ganador(self, puntuaciones_comites: list) -> int:
        if min(puntuaciones_comites) < umbral_actualizacion:
            prediccion = np.argmin(puntuaciones_comites)
        else:
            prediccion = -1
        return prediccion
    
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
    negativos = np.vstack(negativos[:])
    negativos = np.vstack([negativos])
    return negativos