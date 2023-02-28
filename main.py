from agent.agent import Agent
from load_data import load_CORe50
import numpy as np
import datetime
import sys

num_run = int(sys.argv[1])
experiment_name = sys.argv[2]

number_of_subsequences = 15
result_unsup = []
resuls_sup = []
sizes_uns = []
sizes_sup = []

identifier = f"experiments/{experiment_name}/{str(num_run)}"
number_of_experiment: int = 0

__author__ = "Gabriel Vilariño Besteiro"

def experiment():
    print("STARTING EXPERIMENT...")
    first_scene = dataset[0]
    agent = Agent(first_scene, name=identifier) # + "/" + str(number_of_experiment)
    
    test = dataset[9:]
    res_unsup, res_sup = [], []
    size_unsup, size_sup = [], []

    nosup, sup, s_unsup, s_sup = fase_test(agent, test)
    res_unsup.append(nosup)
    res_sup.append(sup)
    size_unsup.append(s_unsup)
    size_sup.append(s_sup)

    for scene in dataset[1:9]:                         # Escenas de entrenamiento, las de test las cargamos siempre como las 3 últimas
        # Fase Entrenamiento
        for id_object, sequence in enumerate(scene):
            subsequences = np.array_split(sequence, number_of_subsequences)
            for subsequence in subsequences:
                agent.train(subsequence, id_object)
        # agent.healing() This is only for unsupervised
        nosup, sup, s_unsup, s_sup = test_phase(agent, test)
        res_unsup.append(nosup)
        res_sup.append(sup)
        size_unsup.append(s_unsup)
        size_sup.append(s_sup)
    print("END OF THE EXPRIMENT")
    result_unsup.append(res_unsup)
    resuls_sup.append(res_sup)
    sizes_uns.append(size_unsup)
    sizes_sup.append(size_sup)
    return agent


def test_phase(agent: Agent, test: list):
    # Test phase
    number_of_sequences_processed = 0
    hits_uns = 0
    hits_sup = 0
    for scene_number in range(len(test)):
        for id_object, test_sequence in enumerate(test[scene_number]):
            subsequences = np.array_split(test_sequence, number_of_subsequences)
            for subsequence in subsequences:
                number_of_sequences_processed +=1
                prediccion_nosup, prediccion_sup = agent.test(subsequence)
                hits_uns += prediccion_nosup == id_object
                hits_sup += prediccion_sup == id_object
    print("\t STEP COMPLETED")
    print("\t\tSequences proccesed: ", number_of_sequences_processed)
    print("\t\tHits uns: ", hits_uns)
    print("\t\tHits sup: ", hits_sup)
    print("\t\tAcc uns: ", float(hits_uns)/number_of_sequences_processed)
    print("\t\tAcc sup: ", float(hits_sup)/number_of_sequences_processed)
    size_unsup, size_sup = agent.ensembles_size()
    print("\t\tMean size of ensembles (uns): ", size_unsup)
    print("\t\tMean size of ensembles (sup): ", size_sup)
    return  round(float(hits_uns)/number_of_sequences_processed, 4), round(float(hits_sup)/number_of_sequences_processed, 4), size_unsup, size_sup

if num_run == 0:
    run_0 = ["s11", "s4", "s2", "s9", "s1", "s6", "s5", "s8", "s3", "s7", "s10"]
    dataset = load_CORe50('../dataset/Core50', run_0)
    agent = experiment()
elif num_run == 1:
    run_1 = ["s2", "s9", "s1", "s8", "s4", "s5", "s11", "s6", "s3", "s7", "s10"]
    dataset = load_CORe50('../dataset/Core50', run_1)
    agent = experiment()
elif num_run == 2:
    run_2 = ["s8", "s2", "s6", "s5", "s4", "s1", "s9", "s11", "s3", "s7", "s10"]
    dataset = load_CORe50('../dataset/Core50', run_2)
    agent = experiment()
elif num_run == 3:
    run_3 = ["s1", "s9", "s2", "s8", "s6", "s11", "s5", "s4", "s3", "s7", "s10"]
    dataset = load_CORe50('../dataset/Core50', run_3)
    agent = experiment()
elif num_run == 4:
    run_4 = ["s5", "s1", "s4", "s8", "s11", "s9", "s6", "s2", "s3", "s7", "s10"]
    dataset = load_CORe50('../dataset/Core50', run_4)
    agent = experiment()
elif num_run == 5:
    run_5 = ["s4", "s5", "s11", "s8", "s2", "s1", "s9", "s6", "s3", "s7", "s10"]
    dataset = load_CORe50('../dataset/Core50', run_5)
    agent = experiment()
elif num_run == 6:
    run_6 = ["s8", "s11", "s1", "s9", "s2", "s4", "s6", "s5", "s3", "s7", "s10"]
    dataset = load_CORe50('../dataset/Core50', run_6)
    agent = experiment()
elif num_run == 7:
    run_7 = ["s6", "s2", "s5", "s8", "s11", "s1", "s9", "s4", "s3", "s7", "s10"]
    dataset = load_CORe50('../dataset/Core50', run_7)
    agent = experiment()
elif num_run == 8:
    run_8 = ["s4", "s5", "s6", "s11", "s2", "s1", "s9", "s8", "s3", "s7", "s10"]
    dataset = load_CORe50('../dataset/Core50', run_8)
    agent = experiment()
elif num_run == 9:
    run_9 = ["s1", "s8", "s9", "s4", "s6", "s2", "s11", "s5", "s3", "s7", "s10"]
    dataset = load_CORe50('../dataset/Core50', run_9)
    agent = experiment()


# Save results
f = open(str(identifier) + "results_unsup.txt", "w")
for exp in result_unsup:
    f.write('\t'.join([str(e) for e in exp]).replace(".", ",") + "\n")
f.write("\n" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
f.write("\n" + str(agent))
f.close()

f = open(str(identifier) + "results_sup.txt", "w")
for exp in resuls_sup:
    f.write('\t'.join([str(e) for e in exp]).replace(".", ",") + "\n")
f.write("\n" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
f.write("\n" + str(agent))
f.close()

f = open(str(identifier) + "ensemble_size_unsup.txt", "w")
for exp in sizes_uns:
    f.write('\t'.join([str(e) for e in exp]).replace(".", ",") + "\n")
f.write("\n" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
f.write("\n" + str(agent))
f.close()

f = open(str(identifier) + "ensemble_size_sup.txt", "w")
for exp in sizes_sup:
    f.write('\t'.join([str(e) for e in exp]).replace(".", ",") + "\n")
f.write("\n" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
f.write("\n" + str(agent))
f.close()
