from sistema.sistema import Sistema
from cargar_datos import cargar_CORe50
import numpy as np
import uuid
import datetime
import os

num_subsecuencias = 30
resultados_nosup = []
resultados_sup = []

identificador = str(uuid.uuid4())
print("IDENTIFICADOR: " + identificador)
os.makedirs(identificador)
num_experimento: int = 0

def experimento():
    print("INICIO EXPERIMENTO")
    primera_escena_inicializacion = dataset[0]
    global num_experimento
    os.makedirs(identificador + "/" + str(num_experimento))
    sistema = Sistema(primera_escena_inicializacion, nombre=identificador + "/" + str(num_experimento))
    num_experimento += 1
    test = dataset[9:]
    res_nosup, res_sup = [], []

    nosup, sup = fase_test(sistema, test)
    res_nosup.append(nosup)
    res_sup.append(sup)

    for escena in dataset[1:9]:                         # Escenas de entrenamiento, las de test las cargamos siempre como las 3 últimas
        # Fase Entrenamiento
        for individuo, secuencia in enumerate(escena):
            secuencias = np.array_split(secuencia, num_subsecuencias)
            for secuencia in secuencias:
                sistema.entrenar(secuencia, individuo)
        sistema.healing()
        nosup, sup = fase_test(sistema, test)
        res_nosup.append(nosup)
        res_sup.append(sup)
    print("FIN EXPERIMENTO")
    resultados_nosup.append(res_nosup)
    resultados_sup.append(res_sup)
    return sistema


def fase_test(sistema: Sistema, test: list):
    # Fase Test
    secuencias_evaluadas = 0
    aciertos_nosup = 0
    aciertos_sup = 0
    for num_escena in range(len(test)):
        for individuo, secuencia_entrenamiento in enumerate(test[num_escena]):
            secuencias = np.array_split(secuencia_entrenamiento, num_subsecuencias)
            for secuencia in secuencias:
                secuencias_evaluadas +=1
                prediccion_nosup, prediccion_sup = sistema.test(secuencia)
                aciertos_nosup += prediccion_nosup == individuo
                aciertos_sup += prediccion_sup == individuo
    print("\tCICLO COMPLETADO")
    print("\t\tSecuencias evaluadas: ", secuencias_evaluadas)
    print("\t\tAciertos nosup: ", aciertos_nosup)
    print("\t\tAciertos sup: ", aciertos_sup)
    print("\t\tPrecision nosup: ", float(aciertos_nosup)/secuencias_evaluadas)
    print("\t\tPrecision sup: ", float(aciertos_sup)/secuencias_evaluadas)
    return  round(float(aciertos_nosup)/secuencias_evaluadas, 4), round(float(aciertos_sup)/secuencias_evaluadas, 4)


run_0 = ["s11", "s4", "s2", "s9", "s1", "s6", "s5", "s8", "s3", "s7", "s10"]
dataset = cargar_CORe50('../caracteristicas/Core50', run_0)
experimento()

run_1 = ["s2", "s9", "s1", "s8", "s4", "s5", "s11", "s6", "s3", "s7", "s10"]
dataset = cargar_CORe50('../caracteristicas/Core50', run_1)
experimento()

run_2 = ["s8", "s2", "s6", "s5", "s4", "s1", "s9", "s11", "s3", "s7", "s10"]
dataset = cargar_CORe50('../caracteristicas/Core50', run_2)
experimento()

run_3 = ["s1", "s9", "s2", "s8", "s6", "s11", "s5", "s4", "s3", "s7", "s10"]
dataset = cargar_CORe50('../caracteristicas/Core50', run_3)
experimento()

run_4 = ["s5", "s1", "s4", "s8", "s11", "s9", "s6", "s2", "s3", "s7", "s10"]
dataset = cargar_CORe50('../caracteristicas/Core50', run_4)
experimento()

run_5 = ["s4", "s5", "s11", "s8", "s2", "s1", "s9", "s6", "s3", "s7", "s10"]
dataset = cargar_CORe50('../caracteristicas/Core50', run_5)
experimento()

run_6 = ["s8", "s11", "s1", "s9", "s2", "s4", "s6", "s5", "s3", "s7", "s10"]
dataset = cargar_CORe50('../caracteristicas/Core50', run_6)
experimento()

run_7 = ["s6", "s2", "s5", "s8", "s11", "s1", "s9", "s4", "s3", "s7", "s10"]
dataset = cargar_CORe50('../caracteristicas/Core50', run_7)
experimento()

run_8 = ["s4", "s5", "s6", "s11", "s2", "s1", "s9", "s8", "s3", "s7", "s10"]
dataset = cargar_CORe50('../caracteristicas/Core50', run_8)
experimento()

run_9 = ["s1", "s8", "s9", "s4", "s6", "s2", "s11", "s5", "s3", "s7", "s10"]
dataset = cargar_CORe50('../caracteristicas/Core50', run_9)
sistema = experimento()


# Guardar resultados
f = open(str(identificador) + "resultados_nosup.txt", "w")
for exp in resultados_nosup:
    f.write('\t'.join([str(e) for e in exp]).replace(".", ",") + "\n")
f.write("\n" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
f.write("\n" + str(sistema))
f.close()

f = open(str(identificador) + "resultados_sup.txt", "w")
for exp in resultados_sup:
    f.write('\t'.join([str(e) for e in exp]).replace(".", ",") + "\n")
f.write("\n" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
f.write("\n" + str(sistema))
f.close()
