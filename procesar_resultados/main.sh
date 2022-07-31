#!/bin/bash

cd ../experimentos/$1
for file in *nosup.txt; 
do
    head -n 1 $file
    head -n 1 $file >> ../procesar_resultados/nosup_$1.txt
done

for file in *_sup.txt; 
do
    head -n 1 $file
    head -n 1 $file >> ../procesar_resultados/sup_$1.txt
done