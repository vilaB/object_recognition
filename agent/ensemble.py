import numpy as np
from agent.SVM import SVM
from agent.tools import number_of_positives, generate_negatives, number_of_negatives

limit_mode = 'div_1'

__author__ = "Gabriel Vilariño Besteiro"

class Ensemble():
    members: list[dict[str, SVM | list | int ]] = None
    name: str = None
    agent = None

    def __init__(self, positives: list, negatives: list, number_of_positives: int, number_of_negatives: int, name: str = None, agent = None) -> None:
        data, tags = build_train_data(positives, negatives, number_of_positives, number_of_negatives)
        svm = SVM(features=data, tags=tags)
        self.members = []
        self.members.append({'member': svm, 'positives': positives, "veces": 0, "useful": 0}) 
        self.name = name
        self.agent = agent

    
    def __str__(self) -> str:
        return self.name
    

    def size_ensemble(self) -> int:
        return len(self.members)

    
    def train(self, positives: list, negatives: list, number_of_positives: int, number_of_negatives: int) -> None:
        data, tags = build_train_data(positives, negatives, number_of_positives, number_of_negatives)
        svm = SVM(features=data, tags=tags)
        self.members.append({'member': svm, 'positives': positives, "veces": 0, "useful": 0})
    

    def process_sequence(self, sequence: list, test: bool = False) -> list:
        matrix = []
        for miembro in self.members:
            prediccion = miembro['member'].process_image(sequence)
            miembro['last_predictions'] = prediccion
            matrix.append(prediccion)                          # Each row is one member, each column one image
        return matrix


    def set_utility(self, score: float):
        for member in self.members:
            if member.get("last_predictions") is not None:
                member["veces_util"] += sum(member['last_predictions'] < score)
                member["veces"] += len(member['last_predictions'])
    

    def get_positives(self) -> list:
        return [member['positives'] for member in self.members]

    
    def mark_member_for_delete(self, member_index: int) -> None:
        if member_index == 0:
            print('ERROR| Can not delete first member of ensemble')
            return
        if member_index == len(self.members) - 1:
            print('AVISO| Deleting last member of ensemble (last one added')
        self.members[member_index]['deleted'] = True


    def delete_marked_members(self) -> None:
        self.members = [miembro for miembro in self.members if miembro.get('deleted') is None]


    def purge_ensembles(self, size: int, init: list) -> None:
        size_bank = 50

        if len(self.members) > size:
            if limit_mode == 'rand':
                to_pop = list(np.random.randint(0, len(self.members), size=len(self.members) - size))
            elif limit_mode in ['div_1', 'div_2', 'gabriel']:
                array = []
                for objeto in init:
                    for matriz_frame in objeto:
                        array.append(matriz_frame)
                # lst = self.get_positives()
                # array = []
                # for positives in lst:
                #     for positivo in positives:
                #         array.append(positivo)
                init = np.array(array)
                init = np.vstack(init)
                negatives = np.vstack([init])
                negatives = pick_random_negatives(negatives, 1000)
                data_bank_red = pick_random_negatives(negatives, size_bank)
                data_bank_red = data_bank_red.astype(np.float32)
                scores = self.process_sequence(data_bank_red)  # [scores_svm_1, scores_svm2 ...] -> ( num_members x num_frames)
                scores = np.transpose(scores)  # (num_frames x num_members)

                if limit_mode == 'div_1':
                    signed_scores = np.sign(scores)  #  The `sign` function returns ``-1 if x < 0, 0 if x==0, 1 if x > 0`` ( num_frames x num_members)
                    div_points = np.zeros([1, signed_scores.shape[1]])  # Array with zeros(num_members)
                    for i in range(signed_scores.shape[0]):  # For every frame...
                        aux_signed = np.reshape(np.repeat(signed_scores[i, :], signed_scores.shape[1]), [signed_scores.shape[1], signed_scores.shape[1]])
                        div_points = div_points + np.dot(aux_signed, signed_scores[i, :]) - signed_scores[i,:] * signed_scores[i,:] 
                elif limit_mode == 'div_2':
                    div_points = np.zeros([1, scores.shape[1]])
                    for i in range(scores.shape[0]):
                        mat_scores = np.reshape(np.repeat(scores[i, :], scores.shape[1]),[scores.shape[1], scores.shape[1]])
                        div_points = div_points + np.sum(np.abs(mat_scores - np.transpose(mat_scores)), axis=0)
                elif limit_mode == 'gabriel':
                    div_points = np.zeros([scores.shape[1]])
                    for frame in scores:
                        frame += 100 
                        for num_classificator, classifier_score in enumerate(frame):
                            differences = np.abs(np.delete(frame, [num_classificator]) - frame[num_classificator])
                            diff = np.sum(differences)
                            div_points[num_classificator] += diff
                if limit_mode != "gabriel":
                    args_to_pop = np.argsort(div_points)  # Order by div points for removing the least diverse
                    to_pop = list(args_to_pop[0, size - len(self.members):])
                    if 0 in to_pop:
                        print("INFO| First member can not be deleted by purge")
                        to_pop.remove(0)
                        to_pop.append(args_to_pop[0, (size - len(self.members)) - 1]) 
                    if len(self.members) - 1 in to_pop:
                        print("INFO| Purging member you just added!")
                else:
                    args_to_pop = np.argsort(div_points) 
                    to_pop = list(args_to_pop[size - len(self.members):])
                    if 0 in to_pop:
                        print("INFO| First member can not be deleted by purge")
                        to_pop.remove(0)
                        to_pop.append(args_to_pop[(size - len(self.members)) - 1]) 
                    if len(self.members) - 1 in to_pop:
                        print("INFO| Purging member you just added!")
                

            to_pop = sorted(to_pop, reverse=True)
            for member in to_pop:
                self.members.pop(member)

            
def pick_random_negatives(muestra, numero_de_negatives):
    if muestra.shape[0] >= numero_de_negatives:
        b = muestra[np.random.choice(muestra.shape[0], numero_de_negatives, replace=False), :] # No need for replacement
    else:
        b = muestra[np.random.choice(muestra.shape[0], numero_de_negatives, replace=True), :]  # Need for replacement
    return b


def build_train_data(positives, negatives, number_of_positives, numero_de_negatives):
    if numero_de_negatives > 0:
        muestra_negatives = pick_random_negatives(negatives, numero_de_negatives)
        muestra = np.vstack([muestra_negatives, np.zeros([number_of_positives, negatives.shape[1] ], dtype = np.float32)]) # Ponemos ceros donde irán los positives
        # Create labels for the training of the each exemplar-SVM
        etiquetas_negatives = -np.ones([muestra_negatives.shape[0], 1], dtype= np.int32)
        etiquetas_positives = np.ones([number_of_positives, 1], dtype= np.int32)
        etiquetas = np.vstack([etiquetas_negatives, etiquetas_positives])
    else:
        # Prepare the array that will contain the training data
        muestra = np.vstack([negatives, np.zeros( [number_of_positives, negatives.shape[1] ], dtype = np.float32) ] )
        # Create labels for the training of the each exemplar-SVM
        etiquetas_negatives = -np.ones( [negatives.shape[0], 1], dtype= np.int32 )
        etiquetas_positives = np.ones( [number_of_positives, 1], dtype= np.int32 )
        etiquetas = np.vstack([etiquetas_negatives, etiquetas_positives] )
    muestra[-number_of_positives:, :] = positives
    return muestra, etiquetas