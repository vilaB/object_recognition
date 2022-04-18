import cv2
import numpy as np
from sklearn import svm
import numpy as np

def normalize(output, svm):
    norm = np.linalg.norm(svm.getSupportVectors())
    if norm == 0: 
       return output
    return output / norm

class SVM:
    svm = None
    
    def __init__(self, muestra: list, etiquetas: list) -> None:
        self.svm = cv2.ml.SVM_create()
        self.svm.setType(cv2.ml.SVM_C_SVC)
        self.svm.setKernel(cv2.ml.SVM_LINEAR)
        self.svm.setClassWeights(np.array([1,100], dtype = np.float32))
        self.svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 1000, 1.e-06))
        self.svm.train(muestra, cv2.ml.ROW_SAMPLE, etiquetas)
    

    def procesar_imagen(self, secuencia: list) -> float:
        return normalize(self.svm.predict(secuencia, flags = cv2.ml.STAT_MODEL_RAW_OUTPUT)[1][:, 0], self.svm)



if __name__ == "__main__":
    muestra = np.array([[0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]])
    etiquetas = np.array([0,0,0,0,0,1,1,1])

    clasificador = SVM(muestra, etiquetas)
    print(clasificador.procesar_imagen(muestra))