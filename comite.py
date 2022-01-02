import numpy as np
from SVM import SVM

def elegir_negativos_aleatoriamente(muestra, numero_de_negativos):
    if muestra.shape[0] >= numero_de_negativos:
        b = muestra[np.random.choice(muestra.shape[0], numero_de_negativos, replace=False), :] # No necesitamos reemplazamiento
    else:
        b = muestra[np.random.choice(muestra.shape[0], numero_de_negativos, replace=True), :]  # Necesitamos reemplazamiento
    return b


def construir_muestra_de_entrenamiento(positivos, negativos, numero_positivos, numero_de_negativos):
    if numero_de_negativos > 0:
        muestra_negativos = elegir_negativos_aleatoriamente(negativos, numero_de_negativos)
        muestra = np.vstack([muestra_negativos, np.zeros([numero_positivos, negativos.shape[1] ], dtype = np.float32)]) # Ponemos ceros donde irán los positivos
        # Create labels for the training of the each exemplar-SVM
        etiquetas_negativos = -np.ones([muestra_negativos.shape[0], 1], dtype= np.int32)
        etiquetas_positivos = np.ones([numero_positivos, 1], dtype= np.int32)
        etiquetas = np.vstack([etiquetas_negativos, etiquetas_positivos])
    else:
        # Prepare the array that will contain the training data
        muestra = np.vstack([negativos, np.zeros( [numero_positivos, negativos.shape[1] ], dtype = np.float32) ] )
        # Create labels for the training of the each exemplar-SVM
        etiquetas_negativos = -np.ones( [negativos.shape[0], 1], dtype= np.int32 )
        etiquetas_positivos = np.ones( [numero_positivos, 1], dtype= np.int32 )
        etiquetas = np.vstack([etiquetas_negativos, etiquetas_positivos] )
    muestra[-numero_positivos:, :] = positivos
    return muestra, etiquetas



class Comite():
    clasificadores = []

    def __init__(self, positivos: list, negativos: list, numero_positivos: int, numero_negativos: int, supervisado: bool = False) -> None:
        muestra, etiquetas = construir_muestra_de_entrenamiento(positivos, negativos, numero_positivos, numero_negativos)
        svm = SVM(muestra=muestra, etiquetas=etiquetas)
        self.clasificadores.append({'clasificador': svm, 'positivos': positivos})

    
    def entrenamiento(self, positivos: list, negativos: list, numero_positivos: int, numero_negativos: int) -> None:
        muestra, etiquetas = construir_muestra_de_entrenamiento(positivos, negativos, numero_positivos, numero_negativos)
        svm = SVM(muestra=muestra, etiquetas=etiquetas)
        self.clasificadores.append({'clasificador': svm, 'positivos': positivos})