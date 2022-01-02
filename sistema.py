from comite import Comite
import numpy as np
from comite import construir_muestra_de_entrenamiento

numero_positivos=25
numero_negativos=100
umbral_actualizacion=np.inf


class Sistema():
    comites_no_supervisados = []
    comites_supervisados = []
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
        # Predicción por parte del sistema no supervisado
        ensembles_scr, FDR_scores = self.__presentCandidate(secuencia, IoIensemble, True)
        if min(ensembles_scr) < umbral_actualizacion:
            iUpdate = np.argmin(ensembles_scr)
        else:
            iUpdate = -1

        if iUpdate >= 0:
            FDR_scr = FDR_scores[iUpdate]
            array_FDR_scr = np.array(FDR_scr)
            index_list = np.argsort(-array_FDR_scr)
            positiveBatch = []
            for index in index_list[:numero_positivos]:
                positiveBatch.append(secuencia[index, :].reshape(1, -1))
            index = index_list[numero_positivos - 1]
            if len(index_list) < numero_positivos:
                for k in range(len(index_list), numero_positivos):
                    positiveBatch.append(secuencia[index, :].reshape(1, -1))
            positiveBatch = np.vstack(positiveBatch)

            negativos = generar_negativos(secuencia, individuo)
            self.comites_no_supervisados[iUpdate].entrenamiento(positiveBatch, negativos, numero_positivos, numero_negativos)
            return iUpdate



def generar_negativos(muestras_inicializacion: list, posicion_positivo: int):
    negativos = np.array(muestras_inicializacion[0:posicion_positivo] + muestras_inicializacion[posicion_positivo + 1:]) # Cogemos como negativos todas las demás secuencias menos la propa: usamos esta aritmética de listas para evitar hacer una deepcopy
    negativos = np.vstack(negativos[:, 0])
    negativos = np.vstack([negativos])
    return negativos