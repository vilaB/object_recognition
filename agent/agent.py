from agent.ensemble import Ensemble
import numpy as np
from scipy import stats
from agent.constants import FDR_FUNCTION, SDR_FUNCTION, WINNR_ENSEMBLE
from agent.tools import generate_negatives, number_of_positives, number_of_negatives, threshold_ack, FDR_function, percentile_FDR, SDR_mode, percentie_SDR, max_size_ensemble, winner_decision, FDR, SDR, purger_supervided, threshold_we_already_ack_this


__author__ = "Gabriel Vilariño Besteiro"


class Agent():
    ensembles_uns: list[Ensemble] = None
    ensembles_sup: list[Ensemble] = None 
    init: list = None
    name: str = None 

    def __init__(self, init: list, ensemble_names: list = None, name: str = None):
        print("Building agent...")
        self.init = init
        self.ensembles_uns = []
        self.ensembles_sup = []
        self.name = name + "/" if name else ""
        for individuo in range(len(init)):
            negatives = generate_negatives(init, individuo)
            ens_uns = Ensemble(positives=init[individuo], negatives=negatives, number_of_positives=number_of_positives, number_of_negatives=number_of_negatives, sistema=self)
            ens_sup = Ensemble(positives=init[individuo], negatives=negatives, number_of_positives=number_of_positives, number_of_negatives=number_of_negatives, sistema=self)
            self.ensembles_uns.append(ens_uns)
            self.ensembles_sup.append(ens_sup)
        if ensemble_names is not None:
            if len(init) != len(ensemble_names):
                raise Exception("The number of ensemble names and init sequences have to match.")
            for i, ens_name in enumerate(ensemble_names):
                self.ensembles_uns[i].name = self.name + ens_name
                self.ensembles_sup[i].name = self.name + ens_name
        else:
            for i in range(len(init)):
                self.ensembles_uns[i].name = self.name + str(i)
                self.ensembles_sup[i].name = self.name + str(i)
        print("Agent built completed.")
        print("Agent parameters are: ")
        print("\t- Number of positives (SVM creation): ", number_of_positives)
        print("\t- Number of negatives (SVM creation): ", number_of_negatives)
        print("\t- Threshold for recognize: ", threshold_ack)
        print("\t- Threshold for learning: ", threshold_we_already_ack_this)
        print("\t- FDR function: ", FDR_function)
        print("\t- FDR percentile: ", percentile_FDR)
        print("\t- SDR function: ", SDR_mode)
        print("\t- SDR percentile: ", percentie_SDR)
        print("\t- Max ensemble size: ", max_size_ensemble)
        print("\t- Winner decision: ", winner_decision)
        print("\t- Purge supervised: ", purger_supervided)

    
    # str method
    def __str__(self):
        return """
        Agent:
            - Number of positives (SVM creation): {}
            - Number of negatives (SVM creation): {}
            - Threshold for recognize: {}
            - FDR function: {}
            - FDR percentile: {}
            - SDR function: {}
            - SDR percentile: {}
            - Max ensemble size: {}
            - Winner decision: {}
            """.format(number_of_positives, number_of_negatives, threshold_ack, FDR_function, percentile_FDR, SDR_mode, percentie_SDR, max_size_ensemble, winner_decision)


    def ensembles_size(self):
        tam_nosup = 0
        for comite in self.ensembles_uns:
            tam_nosup += comite.size_ensemble()
        tam_nosup = tam_nosup / len(self.ensembles_uns)

        tam_sup = 0
        for comite in self.ensembles_sup:
            tam_sup += comite.size_ensemble()
        tam_sup = tam_sup / len(self.ensembles_sup)
        return tam_nosup, tam_sup
        

    def test(self, secuencia: list):
        puntuaciones_ensembles_uns, _ = self.__show_sequence(secuencia, self.ensembles_uns)
        prediccion_no_supervisados = self.__winner_decision(puntuaciones_ensembles_uns) 

        puntuaciones_ensembles_sup, _ = self.__show_sequence(secuencia, self.ensembles_sup)
        prediccion_supervisados = self.__winner_decision(puntuaciones_ensembles_sup)
        return prediccion_no_supervisados, prediccion_supervisados


    def train(self, secuencia: list, individuo: int, supervisar_no_supervisados: bool = False) -> int:
        pred_supervisado = self.entrenamiento_supervisado(secuencia, individuo)
        if purger_supervided and pred_supervisado >= 0:
            self.ensembles_sup[individuo].purge_ensembles(max_size_ensemble, self.init)
        return pred_supervisado

    
    def entrenamiento_no_supervisado(self, secuencia: list):
        # Predicción por parte del sistema no supervisado
        puntuaciones_comites, puntuaciones_imagenes_de_comites = self.__show_sequence(secuencia, self.ensembles_uns)
        prediccion = self.__winner_decision(puntuaciones_comites)
        puntuacion_ganadora = puntuaciones_comites[prediccion]

        if prediccion >= 0: 
            if puntuacion_ganadora < threshold_we_already_ack_this: # Ya se reconoce bien la secuencia
                print(f"INFO - NOSUP|\t Se omite introducir un nuevo miembro en el comité porque la puntuación del comité es {puntuacion_ganadora}, lo que significa que ya lo reconoce con seguridad")
                return -1
            puntuaciones_imagenes = puntuaciones_imagenes_de_comites[prediccion]
            puntuaciones_imagenes = np.array(puntuaciones_imagenes)
            puntuaciones_imagenes = np.absolute(puntuaciones_imagenes)
            indices_ordenados = np.argsort(puntuaciones_imagenes)                       # Nos devuelve una lista con las posiciones con las puntuaciones más bajas (+ cercanas a la frontera del conocimiento)
            positives = []
            f = open("puntuaciones-nuevo-miembro.csv", "a")
            for indice in indices_ordenados[:number_of_positives]:
                if indice != indices_ordenados[0]:
                    f.write(", ")
                f.write(str(puntuaciones_imagenes[indice]))
                positives.append(secuencia[indice, :].reshape(1, -1))
            f.write("\n")
            f.close()
            # indice = indices_ordenados[number_of_positives - 1]
            positives = np.vstack(positives)
            negatives = generate_negatives(self.init, prediccion)
            self.ensembles_uns[prediccion].train(positives, negatives, number_of_positives, number_of_negatives)

            # Para medición de utilidad
            # self.ensembles_uns[prediccion].set_utility(puntuacion_ganadora)
        return prediccion
        
    

    def entrenamiento_supervisado(self, secuencia: list[np.array], individuo: int, comite: Ensemble = None):
        if individuo >= 0:
            if comite is None: comite = self.ensembles_sup[individuo]
            matriz_del_comite = comite.process_sequence(secuencia)  # Devolve unha lista coa puntuación que lle da cada un dos ensembles do IoI
            puntuaciones_imagenes = FDR(matriz_del_comite)  # Calcula la puntuación final, por ejemplo, con la media de la lista
            puntuacion_del_comite = SDR(puntuaciones_imagenes)
            if puntuacion_del_comite < threshold_we_already_ack_this:
                print(f"INFO - SUP|\t Se omite introducir un nuevo miembro en el comité {individuo} porque la puntuación del comité es {puntuacion_del_comite}, lo que significa que ya lo reconoce con seguridad")
                return -1

            puntuaciones_imagenes = np.array(puntuaciones_imagenes)
            puntuaciones_imagenes = np.absolute(puntuaciones_imagenes)
            indices_ordenados = np.argsort(puntuaciones_imagenes)                       # Nos devuelve una lista con las posiciones con las puntuaciones más bajas (+ cercanas a la frontera del conocimiento)
            positives = []
            for indice in indices_ordenados[:number_of_positives]:
                positives.append(secuencia[indice, :].reshape(1, -1))
            # indice = indices_ordenados[number_of_positives - 1]
            positives = np.vstack(positives)
            negatives = generate_negatives(self.init, individuo)
            comite.train(positives, negatives, number_of_positives, number_of_negatives)
            return individuo
    

    def healing(self):
        for i, comite in enumerate(self.ensembles_uns):
            secuencias_positivas = comite.get_positives()
            for j, secuencia in enumerate(secuencias_positivas):                                    # Número de clasificador, secuencia usada para su creación
                puntuaciones, _ = self.__show_sequence(secuencia, self.ensembles_uns)
                prediccion = self.__winner_decision(puntuaciones)
                if prediccion != i:
                    comite.mark_member_for_delete(j)
        for comite in self.ensembles_uns:
            comite.delete_marked_members()


    def __winner_decision(self, puntuaciones_comites: list) -> int:
        if winner_decision == WINNR_ENSEMBLE.THE_BEST:
            if np.min(puntuaciones_comites) < threshold_ack:
                prediccion = np.argmin(puntuaciones_comites)
            else:
                prediccion = -1
        elif winner_decision == WINNR_ENSEMBLE.WEIBULL:
            puntuaciones_comites_ordenadas = np.sort(puntuaciones_comites)
            mayores_respuestas = puntuaciones_comites_ordenadas[1:int(len(puntuaciones_comites_ordenadas)/2)]
            mediana = np.median(puntuaciones_comites_ordenadas[1:])

            distancia = np.abs(mayores_respuestas - mediana)
            puntuacion_ganador = np.abs(puntuaciones_comites_ordenadas[0] - mediana)
            
            shape,_,escala = stats.weibull_min.fit(distancia, floc=0)
            # Decision
            if ( self.__weibull(puntuacion_ganador, escala, shape) < threshold_ack):
                prediccion = np.argmin(puntuaciones_comites)
            else:
                prediccion = -1

        return prediccion

    def __weibull(self, x,n,a):
        return (a / n) * (x / n)**(a - 1) * np.exp(-(x / n)**a)
    
    def __show_sequence(self, secuencia, comites: list[Ensemble]) -> tuple:
        puntuaciones_de_cada_comite = []
        puntuaciones_imagenes_de_comites = []
        for i, comite in enumerate(comites):
            matriz_del_comite = comite.process_sequence(secuencia)  # Devolve unha lista coa puntuación que lle da cada un dos ensembles do IoI
            puntuaciones_imagenes = FDR(matriz_del_comite)  # Calcula la puntuación final, por ejemplo, con la media de la lista
            puntuaciones_imagenes_de_comites.append(puntuaciones_imagenes)
            puntuacion_del_comite = SDR(puntuaciones_imagenes)
            puntuaciones_de_cada_comite.append(puntuacion_del_comite)
        return puntuaciones_de_cada_comite, puntuaciones_imagenes_de_comites

