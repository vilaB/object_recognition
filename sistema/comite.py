from re import S
import numpy as np
from sistema.SVM import SVM
from sistema.tools import generar_negativos, numero_negativos, FDR
 
modo_limitacion = 'div_1'

class Comite():
    miembros: list[dict[str, SVM | list | int ]] = None
    nombre: str = None
    sistema = None

    def __init__(self, positivos: list, negativos: list, numero_positivos: int, numero_negativos: int, nombre: str = None, sistema = None) -> None:
        muestra, etiquetas = construir_muestra_de_entrenamiento(positivos, negativos, numero_positivos, numero_negativos)
        svm = SVM(muestra=muestra, etiquetas=etiquetas)
        self.miembros = []
        self.miembros.append({'clasificador': svm, 'positivos': positivos, "veces": 0, "veces_util": 0})                                # TODO: esto tiene que ser un tipo propio
        self.nombre = nombre
        self.sistema = sistema

    
    def __str__(self) -> str:
        return self.nombre

    
    def entrenamiento(self, positivos: list, negativos: list, numero_positivos: int, numero_negativos: int) -> None:
        muestra, etiquetas = construir_muestra_de_entrenamiento(positivos, negativos, numero_positivos, numero_negativos)
        svm = SVM(muestra=muestra, etiquetas=etiquetas)
        self.miembros.append({'clasificador': svm, 'positivos': positivos, "veces": 0, "veces_util": 0})
    

    def procesar_secuencia(self, secuencia: list, test: bool = False) -> list:
        matriz_puntuaciones = []
        for miembro in self.miembros:
            prediccion = miembro['clasificador'].procesar_imagen(secuencia)
            miembro['ultimas_predicciones'] = prediccion
            matriz_puntuaciones.append(prediccion)                          # Cada fila son las predicciones de un miembro, cada columna es una imagen
        return matriz_puntuaciones


    def establecer_utilidad(self, puntuacion: float):
        for miembro in self.miembros:
            if miembro.get('ultimas_predicciones') is not None:
                miembro["veces_util"] += sum(miembro['ultimas_predicciones'] < puntuacion)
                miembro["veces"] += len(miembro['ultimas_predicciones'])
    
    def purgar_comite_por_utilidad(self, tamano: int) -> None:
        if len(self.miembros) > tamano:
            utilidades = []
            for miembro in self.miembros:
                if miembro["veces"] >= 100:                                                         # TODO: poner en variable
                    miembro["utilidad"] = float(miembro["veces_util"]) / float(miembro["veces"])
                    utilidades.append(miembro["utilidad"])
            utilidad_media = np.mean(utilidades)

            # to_pop = []
            # for i, miembro in enumerate(self.miembros):
            #     if miembro.get("utildiad") is not None and miembro["utilidad"] < utilidad_media and i != 0 and miembro["veces"] >= 100:     # TODO: podría no eliminarse ninguno!
            #         to_pop.append(i)
            # for i in reversed(to_pop):
            #     self.miembros.pop(i)

            if len(self.miembros) > tamano:                     # Si no se borra ninguno
                utilidad_mas_baja = 1
                for i, miembro in enumerate(self.miembros):
                    if miembro.get("utilidad") is not None and utilidad_mas_baja > miembro['utilidad']:
                        utilidad_mas_baja = miembro['utilidad']
                        indice_mas_baja = i
                miembro_1 = self.miembros.pop(indice_mas_baja)

                utilidad_mas_baja = 1
                for i, miembro in enumerate(self.miembros):
                    if miembro.get("utilidad") is not None and utilidad_mas_baja > miembro['utilidad']:
                        utilidad_mas_baja = miembro['utilidad']
                        indice_mas_baja = i
                miembro_2 = self.miembros.pop(indice_mas_baja)
                self.__unir_clasificadores(miembro_1, miembro_2)


    def __unir_clasificadores(self, miembro1: dict, miembro2: dict) -> None:
        postitivos_1 = miembro1['positivos']
        postitivos_2 = miembro2['positivos']
        matriz_1 = miembro1['clasificador'].procesar_imagen(np.array(postitivos_1))
        matriz_2 = miembro2['clasificador'].procesar_imagen(np.array(postitivos_2))
        indices_ordenados_1 = np.argsort(matriz_1)[:int(len(postitivos_1)/2)] 
        indices_ordenados_2 = np.argsort(matriz_2)[:int(len(postitivos_2)/2)]
        
        positivos = []
        for i in indices_ordenados_1:
            positivos.append(postitivos_1[i])
            positivos.append(postitivos_2[i])
        
        numero_comite = int(self.nombre.split("/" )[-1])
        negativos = generar_negativos(self.sistema.muestra_de_inicializacion, numero_comite)
        self.entrenamiento(np.array(positivos), negativos, len(positivos), numero_negativos)





    
    def obtener_positivos(self) -> list:
        return [miembro['positivos'] for miembro in self.miembros]  # Positivos usados para crear cada uno de los clasificadores

    
    def marcar_miembro_para_eliminar(self, indice_miembro: int) -> None:
        if indice_miembro == 0:
            print('ERROR| No se puede eliminar el primer miembro del comite')
            return
        if indice_miembro == len(self.miembros) - 1:
            print('AVISO| Eliminando el ultimo miembro del comite')
        self.miembros[indice_miembro]['eliminar'] = True


    def eliminar_miembros_marcados(self) -> None:
        self.miembros = [miembro for miembro in self.miembros if miembro.get('eliminar') is None]


    # Módulo limitación
    def purgar_comite(self, tamano: int) -> None:

        if len(self.miembros) > tamano:
            if modo_limitacion == 'rand':
                to_pop = list(np.random.randint(0, len(self.miembros), size=len(self.miembros) - tamano))
            elif modo_limitacion in ['div_1', 'div_2']:
                positivos = self.obtener_positivos()
                positivos = np.array(positivos)
                positivos = np.vstack(positivos[:, 0])
                puntuaciones = self.procesar_secuencia(positivos)  # [puntuaciones_svm_1, puntuaciones_svm2 ...] -> cada puntuaciones sn las puntuaciones para toda la secuencia ( num_clasificadores x num_frames)
                puntuaciones = np.transpose(puntuaciones)  # Lo traspone (num_frames x num_clasificadores)

                signed_scores = np.sign(puntuaciones)               # Aplica función signal: The `sign` function returns ``-1 if x < 0, 0 if x==0, 1 if x > 0`` ( num_frames x num_clasificadores)
                div_points = np.zeros([1, signed_scores.shape[1]])  # Array con ceros, del tamaño del número de clasificadores (num_clasificadores)
                for i in range(signed_scores.shape[0]):             # Para cada frame... (fila)
                    aux_signed = np.reshape(np.repeat(signed_scores[i, :], signed_scores.shape[1]), [signed_scores.shape[1], signed_scores.shape[1]])
                    div_points = div_points + np.dot(aux_signed, signed_scores[i, :]) - signed_scores[i,:] * signed_scores[i,:]  # Acumula los valores de diversidad para el frame en cuestión con la movida esta

                args_to_pop = np.argsort(div_points)  # Ordena los puntos de diversidad, como los scores están "en negativo" este argsort nos deja como primero al más alto, que sería el más pequeño (menos diverso).
                to_pop = list(args_to_pop[0, tamano - len(self.miembros):])
                if 0 in to_pop:
                    print("NOTIFICACIÓN| El primer clasificador no puede eliminarse por diversidad")
                    to_pop.remove(0)
                    to_pop.append(args_to_pop[0, (tamano - len(self.miembros)) - 1]) # Tenemos que coger uno más desde el principio hasta el final
                if len(self.miembros) - 1 in to_pop:
                    print("AVISO| Se está eliminando el clasificador que se acaba de introducir!")
                

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