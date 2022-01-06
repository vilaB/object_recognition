from sistema.sistema import Sistema
from cargar_datos import cargar_CORe50


dataset = cargar_CORe50('../caracteristicas/Core50')
print(len(dataset))
print(len(dataset[0]))
print(len(dataset[0][0]))

primera_escena = dataset[0]
sistema = Sistema(primera_escena)

for escena in dataset[1:]:
    for individuo, secuencia in enumerate(escena):
        sistema.entrenar(secuencia, individuo)