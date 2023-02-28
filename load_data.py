import numpy as np
from agent.agent import number_of_positives
import os

__author__ = "Gabriel Vilariño Besteiro"

order_for_showing_objects = ['o1', 'o10', 'o11', 'o12', 'o13', 'o14', 'o15', 'o16', 'o17', 'o18', 'o19', 'o2', 'o20', 'o21', 'o22', 'o23', 'o24', 'o25', 'o26', 'o27', 'o28', 'o29', 'o3', 'o30', 'o31', 'o32', 'o33', 'o34', 'o35', 'o36', 'o37', 'o38', 'o39', 'o4', 'o40', 'o41', 'o42', 'o43', 'o44', 'o45', 'o46', 'o47', 'o48', 'o49', 'o50', 'o5', 'o6', 'o7', 'o8', 'o9']

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def load_CORe50(directory: str, scene_order: list) -> list:
    data = []
    print("[", end="", flush=True)
    for scene_name in scene_order:
        print("-", end="", flush=True)
        scene = []
        for obj in order_for_showing_objects: # o1, o2...
            sequence = []
            for image in os.listdir(directory + '/' + scene_name + '/' + obj): # 1.jpg, 2.jpg...
                img = np.fromfile(directory + '/' + scene_name + '/' + obj + '/' + image, dtype=np.float32)
                sequence.append(normalize(img))
            scene.append(sequence)

        init = 0
        if scene_name == scene_order[0]:
            homogeneus_scene = []
            for sequence in scene:
                homogeneus_scene.append(np.array(sequence[:number_of_positives], dtype = np.float32))
            data.append(homogeneus_scene)
            init = number_of_positives

        min_len = 300
        for sequence in scene:
            if len(sequence[init:]) < min_len:
                min_len = len(sequence)
        homogeneus_scene = []
        for secuencia in scene: 
            homogeneus_scene.append(np.array(secuencia[init:min_len], dtype = np.float32))
        data.append(homogeneus_scene)
    print("]")
    return data