from agent.ensemble import Ensemble
import numpy as np
from scipy import stats
from agent.constants import WINNR_ENSEMBLE
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
            ens_uns = Ensemble(positives=init[individuo], negatives=negatives, number_of_positives=number_of_positives, number_of_negatives=number_of_negatives, agent=self)
            ens_sup = Ensemble(positives=init[individuo], negatives=negatives, number_of_positives=number_of_positives, number_of_negatives=number_of_negatives, agent=self)
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
        size_uns = 0
        for ensemble in self.ensembles_uns:
            size_uns += ensemble.size_ensemble()
        size_uns = size_uns / len(self.ensembles_uns)

        size_sup = 0
        for ensemble in self.ensembles_sup:
            size_sup += ensemble.size_ensemble()
        size_sup = size_sup / len(self.ensembles_sup)
        return size_uns, size_sup
        

    def test(self, sequence: list):
        score_uns, _ = self.__show_sequence(sequence, self.ensembles_uns)
        pred_uns = self.__winner_decision(score_uns) 

        score_sup, _ = self.__show_sequence(sequence, self.ensembles_sup)
        pred_sup = self.__winner_decision(score_sup)
        return pred_uns, pred_sup


    def train(self, sequence: list, number_of_object: int, supervisar_no_supervisados: bool = False) -> int:
        pred_sup = self.supervised_training(sequence, number_of_object)
        if purger_supervided and pred_sup >= 0:
            self.ensembles_sup[number_of_object].purge_ensembles(max_size_ensemble, self.init)
        return pred_sup
        

    def supervised_training(self, sequence: list[np.array], object_number: int, ensemble: Ensemble = None):
        if object_number >= 0:
            if ensemble is None: ensemble = self.ensembles_sup[object_number]
            matrix = ensemble.process_sequence(sequence)
            scores_images = FDR(matrix)  
            score_ensemble = SDR(scores_images)
            if score_ensemble < threshold_we_already_ack_this:
                print(f"INFO - SUP|\t Not adding new member to ensemble {object_number} bc ensemble score is {score_ensemble}, which means we already recognize the object well")
                return -1

            scores_images = np.absolute(scores_images)
            index_sorted = np.argsort(scores_images)
            positives = []
            for i in index_sorted[:number_of_positives]:
                positives.append(sequence[i, :].reshape(1, -1))

            positives = np.vstack(positives)
            negatives = generate_negatives(self.init, object_number)
            ensemble.train(positives, negatives, number_of_positives, number_of_negatives)
            return object_number
    

    # Not used for supervised
    def healing(self):
        for i, ensemble in enumerate(self.ensembles_uns):
            seq_positives = ensemble.get_positives()
            for j, seq in enumerate(seq_positives):                                    # Número de clasificador, secuencia usada para su creación
                scores, _ = self.__show_sequence(seq, self.ensembles_uns)
                pred = self.__winner_decision(scores)
                if pred != i:
                    ensemble.mark_member_for_delete(j)
        for ensemble in self.ensembles_uns:
            ensemble.delete_marked_members()


    def __winner_decision(self, scores_ensembles: list) -> int:
        if winner_decision == WINNR_ENSEMBLE.THE_BEST:
            if np.min(scores_ensembles) < threshold_ack:
                pred = np.argmin(scores_ensembles)
            else:
                pred = -1 # Open set
        elif winner_decision == WINNR_ENSEMBLE.WEIBULL:
            scores_ensembles_sorted = np.sort(scores_ensembles)
            higher_responses = scores_ensembles_sorted[1:int(len(scores_ensembles_sorted)/2)]
            median = np.median(scores_ensembles_sorted[1:])

            distance = np.abs(higher_responses - median)
            puntuacion_ganador = np.abs(scores_ensembles_sorted[0] - median)
            
            shape,_,scale = stats.weibull_min.fit(distance, floc=0)
            # Decision
            if ( self.__weibull(puntuacion_ganador, scale, shape) < threshold_ack):
                pred = np.argmin(scores_ensembles)
            else:
                pred = -1 # Open set

        return pred

    def __weibull(self, x,n,a):
        return (a / n) * (x / n)**(a - 1) * np.exp(-(x / n)**a)
    
    def __show_sequence(self, sequence, ensembles: list[Ensemble]) -> tuple:
        scores_ensembles = []
        scores_ensembles_images = []
        for ensemble in ensembles:
            matrix = ensemble.process_sequence(sequence)
            scored_images = FDR(matrix) 
            scores_ensembles_images.append(scored_images)
            puntuacion_del_comite = SDR(scored_images)
            scores_ensembles.append(puntuacion_del_comite)
        return scores_ensembles, scores_ensembles_images

