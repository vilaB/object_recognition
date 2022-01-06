import cv2
import numpy as np

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
        return self.svm.predict(secuencia, flags = cv2.ml.STAT_MODEL_RAW_OUTPUT)[1][:, 0]