from sistema.sistema import Sistema
from cargar_datos import cargar_CORe50
import numpy as np
import uuid
import datetime
import random

num_subsecuencias = 3
resultados_nosup = []
resultados_sup = []
resultados_open_set = []

porcentaje_desconocidos = 0.2

identificador = uuid.uuid4()

def metricas_open_set(resultados: dict):
    return f"\t\t{resultados['positivo-positivo']}\t{resultados['positivo-negativo']}\t{resultados['negativo-positivo']}\t{resultados['negativo-negativo']}\t{resultados['positivos-fallados']}"


def elegir_desconocidos(objetos: list, porcentaje: float):
    num_desconocidos = int(len(objetos)*porcentaje)
    lst = random.sample(range(0, 50), num_desconocidos)
    lst.sort(reverse=True)
    return lst


def experimento():
    print("INICIO EXPERIMENTO")
    primera_escena_inicializacion = dataset[0]
    desconocidos = elegir_desconocidos(primera_escena_inicializacion, porcentaje_desconocidos)
    for desconocido in desconocidos:
        primera_escena_inicializacion.pop(desconocido)
    sistema = Sistema(primera_escena_inicializacion)
    test = dataset[9:]
    res_nosup, res_sup, open_set_nosup, open_set_sup = [], [], [], []

    nosup, sup, open_uns, open_sup = fase_test(sistema, test, desconocidos)
    res_nosup.append(nosup)
    res_sup.append(sup)
    open_set_nosup.append(open_uns)
    open_set_sup.append(open_sup)

    for escena in dataset[1:9]:                         # Escenas de entrenamiento, las de test las cargamos siempre como las 3 últimas
        # Fase Entrenamiento
        for individuo, secuencia in enumerate(escena):
            if individuo in desconocidos: individuo = -1
            else:
                for desconocido in desconocidos: 
                    if desconocido < individuo:
                        individuo -= 1
            secuencias = np.array_split(secuencia, num_subsecuencias)
            for secuencia in secuencias:
                sistema.entrenar(secuencia, individuo)
        sistema.healing()
        nosup, sup, open_uns, open_sup = fase_test(sistema, test, desconocidos)
        res_nosup.append(nosup)
        res_sup.append(sup)
        open_set_nosup.append(open_uns)
        open_set_sup.append(open_sup)
    print("FIN EXPERIMENTO")
    resultados_nosup.append(res_nosup)
    resultados_sup.append(res_sup)
    resultados_open_set.append((open_set_nosup, open_set_sup))
    return sistema


def fase_test(sistema: Sistema, test: list, desconocidos: list = None):
    # Fase Test
    secuencias_evaluadas = 0
    aciertos_nosup = 0
    aciertos_sup = 0

    # Para la matriz de OpenSet
    open_set_sup =   {'positivo-positivo': 0, 'positivo-negativo': 0, 'negativo-positivo': 0, 'negativo-negativo': 0, 'positivos-fallados': 0}       # original-prediccion
    open_set_nosup = {'positivo-positivo': 0, 'positivo-negativo': 0, 'negativo-positivo': 0, 'negativo-negativo': 0, 'positivos-fallados': 0}
    for num_escena in range(len(test)):
        for individuo, secuencia_entrenamiento in enumerate(test[num_escena]):
            for desconocido in desconocidos: 
                if desconocido < individuo:
                    individuo -= 1
            secuencias = np.array_split(secuencia_entrenamiento, num_subsecuencias)
            for secuencia in secuencias:
                secuencias_evaluadas +=1
                prediccion_nosup, prediccion_sup = sistema.test(secuencia)
                acierto_nosup = prediccion_nosup == individuo
                acierto_sup = prediccion_sup == individuo
                if desconocidos is not None:
                    if individuo in desconocidos:
                        # No supervisado
                        if acierto_nosup: open_set_nosup['negativo-negativo'] += 1
                        else: open_set_nosup['negativo-positivo'] += 1
                        # Supervisado
                        if acierto_sup: open_set_sup['negativo-negativo'] += 1
                        else: open_set_sup['negativo-positivo'] += 1
                    else:
                        # No supervisado
                        if acierto_nosup: open_set_nosup['positivo-positivo'] += 1
                        else:
                            if prediccion_nosup >= 0:
                                open_set_nosup['positivos-fallados'] += 1
                            else: 
                                open_set_nosup['positivo-negativo'] += 1
                        # Supervisado
                        if acierto_sup: open_set_sup['positivo-positivo'] += 1
                        else:
                            if prediccion_sup >= 0:
                                open_set_sup['positivos-fallados'] += 1
                            else: 
                                open_set_sup['positivo-negativo'] += 1
                aciertos_nosup += acierto_nosup
                aciertos_sup += acierto_sup
    print("\tCICLO COMPLETADO")
    print("\t\tSecuencias evaluadas: ", secuencias_evaluadas)
    print("\t\tAciertos nosup: ", aciertos_nosup)
    print("\t\tAciertos sup: ", aciertos_sup)
    print("\t\tPrecision nosup: ", float(aciertos_nosup)/secuencias_evaluadas)
    print("\t\tPrecision sup: ", float(aciertos_sup)/secuencias_evaluadas)
    return  round(float(aciertos_nosup)/secuencias_evaluadas, 4), round(float(aciertos_sup)/secuencias_evaluadas, 4), open_set_nosup, open_set_sup


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

f = open(str(identificador) + "resultados_open_set.txt", "w")
for i, exp in enumerate(resultados_open_set):
    no_sup, sup = exp
    f.write(f'EXPERIMENTO {i} \n')
    f.write("\tNO SUPERVISADO" + "\n")
    f.write("\t\t++ +- -+ --" + "\n")
    for test in no_sup:
        f.write(metricas_open_set(test) + '\n')
    f.write("\tSUPERVISADO" + "\n")
    f.write("\t\t++ +- -+ --" + "\n")
    for test in sup:
        f.write(metricas_open_set(test) + '\n')
    f.write("\n\n")
f.write("\tRESUMEN\n")
for exp in resultados_open_set:
    no_sup, sup = exp
    no_sup = no_sup[-1]
    sup = sup[-1]
    f.write(metricas_open_set(no_sup) + '\t\t')
    f.write(metricas_open_set(sup) + '\n')
    

f.write("\n" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
f.write("\n" + str(sistema))
f.close()

