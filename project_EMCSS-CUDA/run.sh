#!/bin/bash

make rebuild

echo -e "\n**** Dataset 1 ****"
./main ./dataset/1/S1_3_3.txt ./dataset/1/T1_3750_4320.txt

echo -e "\n**** Dataset 2 ****"
./main ./dataset/2/S2_5_5.txt ./dataset/2/T2_7750_1320.txt

echo -e "\n**** Dataset 3 ****"
./main ./dataset/3/S3_3_3.txt ./dataset/3/T3_8140_9925.txt

echo -e "\n**** Dataset 4 ****"
./main ./dataset/4/S4_5_5.txt ./dataset/4/T4_50_50.txt
