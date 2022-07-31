#!/bin/bash

cd ../experimentos/$1
for file in *.txt; 
do
    head -n 1 $file
    head -n 1 $file >> ../procesar_resultados/$1
done
