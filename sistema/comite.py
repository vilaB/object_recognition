import numpy as np
from sistema.SVM import SVM

 
modo_limitacion = 'div_1'

class Comite():
    miembros: list = None

    def __init__(self, positivos: list, negativos: list, numero_positivos: int, numero_negativos: int, supervisado: bool = False) -> None:
        muestra, etiquetas = construir_muestra_de_entrenamiento(positivos, negativos, numero_positivos, numero_negativos)
        svm = SVM(muestra=muestra, etiquetas=etiquetas)
        self.miembros = []
        self.miembros.append({'clasificador': svm, 'positivos': positivos})

    
    def entrenamiento(self, positivos: list, negativos: list, numero_positivos: int, numero_negativos: int) -> None:
        muestra, etiquetas = construir_muestra_de_entrenamiento(positivos, negativos, numero_positivos, numero_negativos)
        svm = SVM(muestra=muestra, etiquetas=etiquetas)
        self.miembros.append({'clasificador': svm, 'positivos': positivos})
    

    def procesar_secuencia(self, secuencia: list) -> list:
        matriz_puntuaciones = []
        for miembro in self.miembros:
            prediccion = miembro['clasificador'].procesar_imagen(secuencia)
            matriz_puntuaciones.append(prediccion)                          # Cada fila son las predicciones de un miembro, cada columna es una imagen
        return matriz_puntuaciones

    
    def obtener_positivos(self) -> list:
        return [miembro['positivos'] for miembro in self.miembros]  # Positivos usados para crear cada uno de los clasificadores

    
    def marcar_miembro_para_eliminar(self, indice_miembro: int) -> None:
        if indice_miembro == 0:
            print('No se puede eliminar el primer miembro del comite')
            return
        self.miembros[indice_miembro]['eliminar'] = True


    def eliminar_miembros_marcados(self) -> None:
        self.miembros = [miembro for miembro in self.miembros if miembro.get('eliminar') is None]


    # Módulo limitación
    def purgar_comite(self, tamano: int, muestra_inicializacion: list) -> None:
        tamano_banco_div1_2 = 50

        if len(self.miembros) > tamano:
            if modo_limitacion == 'rand':
                to_pop = list(np.random.randint(0, len(self.miembros), size=len(self.miembros) - tamano))
            elif modo_limitacion in ['div_1', 'div_2']:
                muestra_inicializacion = np.array(muestra_inicializacion)
                muestra_inicializacion = np.vstack(muestra_inicializacion[:, 0])
                negativos = np.vstack([muestra_inicializacion])
                negatives = elegir_negativos_aleatoriamente(negativos, 1000)
                data_bank_red = elegir_negativos_aleatoriamente(negatives, tamano_banco_div1_2)
                data_bank_red = data_bank_red.astype(np.float32)
                puntuaciones = self.procesar_secuencia(data_bank_red)  # [puntuaciones_svm_1, puntuaciones_svm2 ...] -> cada puntuaciones sn las puntuaciones para toda la secuencia ( num_clasificadores x num_frames)
                puntuaciones = np.transpose(puntuaciones)  # Lo traspone (num_frames x num_clasificadores)
                if modo_limitacion == 'div_1':
                    signed_scores = np.sign(puntuaciones)  # Aplica función signal: The `sign` function returns ``-1 if x < 0, 0 if x==0, 1 if x > 0`` ( num_frames x num_clasificadores)
                    div_points = np.zeros([1, signed_scores.shape[1]])  # Array con ceros, del tamaño del número de clasificadores (num_clasificadores)
                    for i in range(signed_scores.shape[0]):  # Para cada frame... (fila)
                        aux_signed = np.reshape(np.repeat(signed_scores[i, :], signed_scores.shape[1]), [signed_scores.shape[1], signed_scores.shape[1]])
                        div_points = div_points + np.dot(aux_signed, signed_scores[i, :]) - signed_scores[i,:] * signed_scores[i,:]  # Acumula los valores de diversidad para el frame en cuestión con la movida esta
                    args_to_pop = np.argsort(div_points)  # Ordena los puntos de diversidad, como los scores están "en negativo" este argsort nos deja como primero al más alto, que sería el más pequeño (menos diverso).
                    to_pop = list(args_to_pop[0, tamano - len(self.miembros):])
                elif modo_limitacion == 'div_2':
                    div_points = np.zeros([1, puntuaciones.shape[1]])
                    for i in range(puntuaciones.shape[0]):
                        mat_scores = np.reshape(np.repeat(puntuaciones[i, :], puntuaciones.shape[1]),[puntuaciones.shape[1], puntuaciones.shape[1]])
                        div_points = div_points + np.sum(np.abs(mat_scores - np.transpose(mat_scores)), axis=0)
                    args_to_pop = np.argsort(div_points)
                    to_pop = list(args_to_pop[0, :len(self.miembros) - tamano])

            to_pop = sorted(to_pop, reverse=True)
            for miembro in to_pop:
                self.miembros.pop(miembro)

            





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