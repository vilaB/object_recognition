#!/bin/bash

mkdir "./experiments/$1"
python3 ./main.py 0 $1 > "./experiments/$1/output_0"
python3 ./main.py 1 $1 > "./experiments/$1/output_1"
python3 ./main.py 2 $1 > "./experiments/$1/output_2"
python3 ./main.py 3 $1 > "./experiments/$1/output_3"
python3 ./main.py 4 $1 > "./experiments/$1/output_4"
python3 ./main.py 5 $1 > "./experiments/$1/output_5"
python3 ./main.py 6 $1 > "./experiments/$1/output_6"
python3 ./main.py 7 $1 > "./experiments/$1/output_7"
python3 ./main.py 8 $1 > "./experiments/$1/output_8"
python3 ./main.py 9 $1 > "./experiments/$1/output_9"