#!/bin/bash

python L_rule.py $1 temp.L
python N_rule.py temp.L temp.LN
phthon M_rule.py temp.LN temp.LNM
python R_rule.py temp.LNM temp.LNMR

cat temp.LNMR | sed 's/ NULL//g' | apply_map.pl -f 2- replace_v6.txt | sort -u > new_lexicon.txt
