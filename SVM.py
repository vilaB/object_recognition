import cv2
import numpy as np

class SVM:
    svm = None
    

    def __init__(self, muestra: list, etiquetas: list) -> None:
        svm = cv2.ml.SVM_create()
        svm.setType(cv2.ml.SVM_C_SVC)
        svm.setKernel(cv2.ml.SVM_LINEAR)
        svm.setClassWeights(np.array([1,100], dtype = np.float32))
        svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 1000, 1.e-06))
        svm.train(muestra, cv2.ml.ROW_SAMPLE, etiquetas)  # Adestra cos primeiros 5 frames
    

    def procesar_imagen(self, secuencia: list) -> float:
        return self.predict(secuencia, flags = cv2.ml.STAT_MODEL_RAW_OUTPUT)[1][:,0]