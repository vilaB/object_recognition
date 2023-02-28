import numpy as np
from agent.constants import FDR_FUNCTION, SDR_FUNCTION, WINNR_ENSEMBLE
import statistics

purger_supervided=True
number_of_positives=10
number_of_negatives=100
threshold_ack=np.inf                            # np.inf for closed set
threshold_we_already_ack_this=0                 # Threshold for adding or not a member to the ensemble
FDR_function = FDR_FUNCTION.PERCENTILE          # Frame decision
percentile_FDR = 0.16            
take_number_n = 3                               # For fix mode (take the n)
SDR_mode = SDR_FUNCTION.MEDIAN                  # Sequence decision
percentie_SDR = 0.25
max_size_ensemble = 60
winner_decision = WINNR_ENSEMBLE.THE_BEST       # Mayor respuesta o weibull

__author__ = "Gabriel Vilariño Besteiro"

def generate_negatives(init_images: list, positive_sequence_number: int):
    negatives = np.array(init_images[0:positive_sequence_number] + init_images[positive_sequence_number + 1:]) # Cogemos como negatives todas las demás secuencias menos la propa: usamos esta aritmética de listas para evitar hacer una deepcopy
    negatives = np.vstack(negatives[:])
    negatives = np.vstack([negatives])
    return negatives


# Assign an score to each frame, by deciding between the scores of each member of the ensemble
def FDR(scores):
    if FDR_function == FDR_FUNCTION.MEDIAN:     score = np.median(scores, axis=0)
    elif FDR_function == FDR_FUNCTION.PERCENTILE: score = np.quantile(scores, percentile_FDR, axis=0)
    elif FDR_function == FDR_FUNCTION.THE_BEST:  score = np.min(scores, axis=0)
    elif FDR_function == FDR_FUNCTION.FIX:
        if scores.shape[0] >= take_number_n:
            score = np.sort(scores, axis=0)[2, :]
        else:
            for i in range(1, take_number_n):
                if scores.shape[0] == take_number_n - i:
                    score = np.sort(scores, axis=0)[(take_number_n - i) - 1, :]
                    break
    return score

# To assign a score in the sequence level, we take all the scores for every frame and apply the median or the percentile
def SDR(scores):
    if SDR_mode == SDR_FUNCTION.MEDIAN:         score = statistics.median(scores)
    elif SDR_mode == SDR_FUNCTION.PERCENTILE:   score = np.quantile(scores, percentie_SDR)
    return score