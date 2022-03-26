# object_recognition
Este es el sistema de aprendizaje continuo aplicado al reconocimiento de objetos. Toma como partida OSDe-SVM, una tesis doctoral realizada en la USC. 

# SVM
En SVM.py tenemos la creación de cada uno de los clasificadores del sistema. Se han retirado a un fichero separado de forma que podamos cambiarlo con extrema sencillez.

# Comite
Cada uno de los comites del sistema. Cada comité está formado por una lista de miembros. Esto miembros los representaremos en Python como diccionarios de la forma {'clasificador': svm, 'positivos': positivos}, es bastante intuitivo lo que es clasificador. Por su parte, positivos es la secuencia de positivos utilizada para la creación del clasificador (para el módulo de curación).

# Sistema
Formado por una lista de comités, en realidad dos, una para el modo supervisado y otra para el no supervisado.

# Instalación
Instala con pip cada una de las dependencias. Intenta usar versiones de Python actuales, 3.9, 3.10, pues uso el typing más reciente implementado en estas últimas. Viva el actualizarse!

# Docker
Para lanzar el docker, primero tenderemos que construirlo:
docker build . -t experimento:1.0
Luego ya podemos lanzarlo con:
docker run -d -v /root/EXPERIMENTOS/object_recognition/:/EXPERIMENTOS -v /root/EXPERIMENTOS/dataset/:/dataset -m 4g experimento:1.0

Para ver como va docker logs id --follow

docker build . -t exp:1.0
docker run exp:1.0

# Contacto
Si tienes cualquier duda con este código no dudes en contactarme en gabrivb@outlook.es o en gabriel.vilarino@rai.usc.es. Estaré encantado de echarte una mano.