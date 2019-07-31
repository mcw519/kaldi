#!/bin/bash

python stress2tone.py $1 temp.s2t
python L_rule-v7.py temp.s2t temp.L
python N_rule-v7.py temp.L temp.LN
python M_rule-v7.py temp.LN temp.LNM
python R_rule-v7.py temp.LNM temp.LNMR

cat temp.LNMR | sed 's/ NULL//g' | apply_map.pl -f 2- replace_v7.txt | sort -u > $2

rm temp.*
