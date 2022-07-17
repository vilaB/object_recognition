#!/bin/bash

mkdir "./experimentos/$1"
python3 ./main.py 0 $1 > "./experimentos/$1/salida_0"
python3 ./main.py 1 $1 > "./experimentos/$1/salida_1"
python3 ./main.py 2 $1 > "./experimentos/$1/salida_2"
python3 ./main.py 3 $1 > "./experimentos/$1/salida_3"
python3 ./main.py 4 $1 > "./experimentos/$1/salida_4"
python3 ./main.py 5 $1 > "./experimentos/$1/salida_5"
python3 ./main.py 6 $1 > "./experimentos/$1/salida_6"
python3 ./main.py 7 $1 > "./experimentos/$1/salida_7"
python3 ./main.py 8 $1 > "./experimentos/$1/salida_8"
python3 ./main.py 9 $1 > "./experimentos/$1/salida_9"