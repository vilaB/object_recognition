from sistema.sistema import Sistema
from cargar_datos import cargar_CORe50
import numpy as np

num_subsecuencias = 3
dataset = cargar_CORe50('../caracteristicas/Core50')

primera_escena_inicialziacion = dataset[0]
sistema = Sistema(primera_escena_inicialziacion)
test = dataset[9:]

for escena in dataset[1:9]:                         # Escenas de entrenamiento, las de test las cargamos siempre como las 3 últimas
    # Fase Entrenamiento
    for individuo, secuencia in enumerate(escena):
        secuencias = np.array_split(secuencia, num_subsecuencias)
        for secuencia in secuencias:
            sistema.entrenar(secuencia, individuo)

    # Fase Test
    secuencias_evaluadas = 0
    aciertos_nosup = 0
    aciertos_sup = 0
    for escena_entrenamiento in enumerate(test):
        for individuo, secuencia in enumerate(escena_entrenamiento):
            secuencias = np.array_split(secuencia, num_subsecuencias)
            for secuencia in secuencias:
                secuencias_evaluadas +=1
                acierto_nosup, acierto_sup = sistema.test(secuencia, individuo)
                aciertos_nosup += acierto_nosup
                aciertos_sup += acierto_sup
    print("CICLO COMPLETADO")
    print("\tSecuencias evaluadas: ", secuencias_evaluadas)
    print("\tAciertos nosup: ", aciertos_nosup)
    print("\tAciertos sup: ", aciertos_sup)
    print("\tPrecision nosup: ", float(aciertos_nosup)/secuencias_evaluadas)
    print("\tPrecision sup: ", float(aciertos_sup)/secuencias_evaluadas)