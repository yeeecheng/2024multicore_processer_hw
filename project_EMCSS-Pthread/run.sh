#!/bin/bash

make rebuild

SOURCE=./main

echo -e "\n**** Dataset 1 ****"
for mode in  PCC SSD
do 

    for i in {1..12}
    do 
        $SOURCE ./dataset/1/S1_3_3.txt ./dataset/1/T1_3750_4320.txt $mode $i
    done

    echo -e "\n**** Dataset 2 ****"
    for i in {1..12}
    do
        $SOURCE ./dataset/2/S2_5_5.txt ./dataset/2/T2_7750_1320.txt $mode $i
    done

    echo -e "\n**** Dataset 3 ****"
    for i in {1..12}
    do
        $SOURCE ./dataset/3/S3_3_3.txt ./dataset/3/T3_8140_9925.txt $mode $i
    done

    echo -e "\n**** Dataset 4 ****"
    for i in {1..12}
    do
        $SOURCE ./dataset/4/S4_5_5.txt ./dataset/4/T4_50_50.txt $mode $i
    done
done
