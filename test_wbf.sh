#!/bin/bash
#weights=('DAPNet_144' 'DAPNet_145' 'DAPNet_146' 'DAPNet_147' 'DAPNet_148' 'DAPNet_149' 'DAPNet_140' 'DAPNet_141' 'DAPNet_142' 'DAPNet_143')
weights=('DAPNet_140' 'DAPNet_145' 'DAPNet_150' 'DAPNet_155' 'DAPNet_160')


for weight in "${weights[@]}"
do
    python Run-mean.py -weight "$weight" -seed_a 123 -seed_b 456 -seed_c 789 -exp 0 -lr_mult 10  &
    wait $!
done

for weight in "${weights[@]}"
do
    python Run-mean.py -weight "$weight" -seed_a 123 -seed_b 456 -seed_c 789 -exp 0 -lr_mult 5  &
    wait $!
done

wait
