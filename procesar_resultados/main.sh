#!/bin/bash

cd ../experimentos/$1
touch ../../procesar_resultados/nosup_$1.txt
for file in *resultados_nosup.txt; 
do
    head -n 1 $file
    head -n 1 $file >> ../../procesar_resultados/nosup_$1.txt
done

touch ../../procesar_resultados/sup_$1.txt
for file in *resultados_sup.txt; 
do
    head -n 1 $file
    head -n 1 $file >> ../../procesar_resultados/sup_$1.txt
done