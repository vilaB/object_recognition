from sistema.sistema import Sistema
from cargar_datos import cargar_CORe50
import numpy as np
import uuid
import datetime
import os
import sys

num_run = int(sys.argv[1])
nombre_experimento = sys.argv[2]

num_subsecuencias = 15
resultados_nosup = []
resultados_sup = []
tamanos_nosup = []
tamanos_sup = []

identificador = f"experimentos/{nombre_experimento}/{str(num_run)}"
num_experimento: int = 0

def experimento():
    print("INICIO EXPERIMENTO")
    primera_escena_inicializacion = dataset[0]
    sistema = Sistema(primera_escena_inicializacion, nombre=identificador) # + "/" + str(num_experimento)
    # num_experimento += 1
    test = dataset[9:]
    res_nosup, res_sup = [], []
    tam_nosup, tam_sup = [], []

    nosup, sup, t_nosup, t_sup = fase_test(sistema, test)
    res_nosup.append(nosup)
    res_sup.append(sup)
    tam_nosup.append(t_nosup)
    tam_sup.append(t_sup)

    for escena in dataset[1:9]:                         # Escenas de entrenamiento, las de test las cargamos siempre como las 3 últimas
        # Fase Entrenamiento
        for individuo, secuencia in enumerate(escena):
            secuencias = np.array_split(secuencia, num_subsecuencias)
            for secuencia in secuencias:
                sistema.entrenar(secuencia, individuo)
        sistema.healing()
        nosup, sup, t_nosup, t_sup = fase_test(sistema, test)
        res_nosup.append(nosup)
        res_sup.append(sup)
        tam_nosup.append(t_nosup)
        tam_sup.append(t_sup)
    print("FIN EXPERIMENTO")
    resultados_nosup.append(res_nosup)
    resultados_sup.append(res_sup)
    tamanos_nosup.append(tam_nosup)
    tamanos_sup.append(tam_sup)
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
    tam_nosup, tam_sup = sistema.tamanos_sistema()
    print("\t\tTamaño en modo no supervisado: ", tam_nosup)
    print("\t\tTamaño en modo supervisado: ", tam_sup)
    return  round(float(aciertos_nosup)/secuencias_evaluadas, 4), round(float(aciertos_sup)/secuencias_evaluadas, 4), tam_nosup, tam_sup

if num_run == 0:
    run_0 = ["s11", "s4", "s2", "s9", "s1", "s6", "s5", "s8", "s3", "s7", "s10"]
    dataset = cargar_CORe50('../dataset/Core50', run_0)
    sistema = experimento()
elif num_run == 1:
    run_1 = ["s2", "s9", "s1", "s8", "s4", "s5", "s11", "s6", "s3", "s7", "s10"]
    dataset = cargar_CORe50('../dataset/Core50', run_1)
    sistema = experimento()
elif num_run == 2:
    run_2 = ["s8", "s2", "s6", "s5", "s4", "s1", "s9", "s11", "s3", "s7", "s10"]
    dataset = cargar_CORe50('../dataset/Core50', run_2)
    sistema = experimento()
elif num_run == 3:
    run_3 = ["s1", "s9", "s2", "s8", "s6", "s11", "s5", "s4", "s3", "s7", "s10"]
    dataset = cargar_CORe50('../dataset/Core50', run_3)
    sistema = experimento()
elif num_run == 4:
    run_4 = ["s5", "s1", "s4", "s8", "s11", "s9", "s6", "s2", "s3", "s7", "s10"]
    dataset = cargar_CORe50('../dataset/Core50', run_4)
    sistema = experimento()
elif num_run == 5:
    run_5 = ["s4", "s5", "s11", "s8", "s2", "s1", "s9", "s6", "s3", "s7", "s10"]
    dataset = cargar_CORe50('../dataset/Core50', run_5)
    sistema = experimento()
elif num_run == 6:
    run_6 = ["s8", "s11", "s1", "s9", "s2", "s4", "s6", "s5", "s3", "s7", "s10"]
    dataset = cargar_CORe50('../dataset/Core50', run_6)
    sistema = experimento()
elif num_run == 7:
    run_7 = ["s6", "s2", "s5", "s8", "s11", "s1", "s9", "s4", "s3", "s7", "s10"]
    dataset = cargar_CORe50('../dataset/Core50', run_7)
    sistema = experimento()
elif num_run == 8:
    run_8 = ["s4", "s5", "s6", "s11", "s2", "s1", "s9", "s8", "s3", "s7", "s10"]
    dataset = cargar_CORe50('../dataset/Core50', run_8)
    sistema = experimento()
elif num_run == 9:
    run_9 = ["s1", "s8", "s9", "s4", "s6", "s2", "s11", "s5", "s3", "s7", "s10"]
    dataset = cargar_CORe50('../dataset/Core50', run_9)
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

f = open(str(identificador) + "tamanos_nosup.txt", "w")
for exp in tamanos_nosup:
    f.write('\t'.join([str(e) for e in exp]).replace(".", ",") + "\n")
f.write("\n" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
f.write("\n" + str(sistema))
f.close()

f = open(str(identificador) + "tamanos_sup.txt", "w")
for exp in tamanos_sup:
    f.write('\t'.join([str(e) for e in exp]).replace(".", ",") + "\n")
f.write("\n" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
f.write("\n" + str(sistema))
f.close()
