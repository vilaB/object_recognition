import cv2
import numpy as np
from sklearn import svm

class SVMOpenCV:
    svm = None
    
    def __init__(self, muestra: list, etiquetas: list) -> None:
        self.svm = cv2.ml.SVM_create()
        self.svm.setType(cv2.ml.SVM_C_SVC)
        self.svm.setKernel(cv2.ml.SVM_LINEAR)
        self.svm.setClassWeights(np.array([1,100], dtype = np.float32))
        self.svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 1000, 1.e-06))
        self.svm.train(muestra, cv2.ml.ROW_SAMPLE, etiquetas)
    

    def procesar_imagen(self, secuencia: list) -> float:
        return self.svm.predict(secuencia, flags = cv2.ml.STAT_MODEL_RAW_OUTPUT)[1][:, 0]

    
class SVM:
    svm = None
    
    def __init__(self, muestra: list, etiquetas: list) -> None:
        self.svm = svm.LinearSVC(C=1.e-06)
        self.svm = self.svm.fit(muestra, etiquetas.ravel())
    

    def procesar_imagen(self, secuencia: list) -> float:
        scores = self.svm.decision_function(secuencia)      # Confidence scores per (n_samples, n_classes) combination. In the binary case, confidence score for self.classes_[1] where >0 means this class would be predicted.
        w_norm = np.linalg.norm(self.svm.coef_)
        dist = scores / w_norm                              # Normalizada      
        return -dist



if __name__ == "__main__":
    muestra = np.array([[0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]])
    etiquetas = np.array([0,0,0,0,0,1,1,1])

    clasificador = SVM(muestra, etiquetas)
    print(clasificador.procesar_imagen(muestra))