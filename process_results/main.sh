#!/bin/bash

rm *.txt

echo ""
echo ""
echo "--- Unsupervised mode ---"
cd ../experiments/$1
touch ../../process_results/unsup_$1.txt
for file in *results_unsup.txt; 
do
    head -n 1 $file
    head -n 1 $file >> ../../process_results/unsup_$1.txt
done

echo ""
echo ""
echo "--- Supervised mode ---"
touch ../../process_results/sup_$1.txt
for file in *results_sup.txt; 
do
    head -n 1 $file
    head -n 1 $file >> ../../process_results/sup_$1.txt
done


echo ""
echo ""
echo "--- Unsupervised mode (ensemble size)---"
touch ../../process_results/unsup_$1.txt
for file in *ensemble_size_unsup.txt; 
do
    head -n 1 $file
    head -n 1 $file >> ../../process_results/nosup_$1_ensemble_size.txt
done

echo ""
echo ""
echo "--- Supervised mode (ensemble size)---"
touch ../../process_results/sup_$1.txt
for file in *ensemble_size_sup.txt; 
do
    head -n 1 $file
    head -n 1 $file >> ../../process_results/sup_$1_ensemble_size.txt
done