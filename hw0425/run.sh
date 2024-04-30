#!/bin/bash

TIMES=5;

while read line;
do
	# echo $line;
	read -r n m <<< $line;
	output_file="./res/vecAdd_"$n"_$m.txt";
	echo "Writing result into $output_file";
	for i in $(seq 1 $TIMES);
	  do
	    echo "Times: $i";
	    echo "Times: $i" >> $output_file;
	    ./vec_add $n $m >> $output_file;
	  done
done < ./data.txt

