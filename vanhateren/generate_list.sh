#!/bin/bash

NUM_TRAIN=3500

rm -f ordered_list
for file in *.png; do
    echo $(realpath $file) >> ordered_list 
done

sort -R ordered_list > shuffled_list
head -n $NUM_TRAIN shuffled_list > training_list
tail -n $(( $NUM_TRAIN + 1 )) shuffled_list > testing_list

