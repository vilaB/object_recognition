#!/bin/bash

echo ""
echo ""
echo "--- Modo no supervisado ---"
cd ../experimentos/$1
touch ../../procesar_resultados/nosup_$1.txt
for file in *resultados_nosup.txt; 
do
    head -n 1 $file
    head -n 1 $file >> ../../procesar_resultados/nosup_$1.txt
done

echo ""
echo ""
echo "--- Modo supervisado ---"
touch ../../procesar_resultados/sup_$1.txt
for file in *resultados_sup.txt; 
do
    head -n 1 $file
    head -n 1 $file >> ../../procesar_resultados/sup_$1.txt
done


echo ""
echo ""
echo "--- Modo no supervisado (tamaños)---"
cd ../experimentos/$1
touch ../../procesar_resultados/nosup_$1.txt
for file in *tamanos_nosup.txt; 
do
    head -n 1 $file
    head -n 1 $file >> ../../procesar_resultados/nosup_$1_tamanos.txt
done

echo ""
echo ""
echo "--- Modo supervisado (tamaños)---"
touch ../../procesar_resultados/sup_$1.txt
for file in *tamanos_sup.txt; 
do
    head -n 1 $file
    head -n 1 $file >> ../../procesar_resultados/sup_$1_tamanos.txt
done