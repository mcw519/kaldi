#!/bin/bash

arpa=$1
lang=$2

if [ $# != 2 ]; then
    echo "Usage: build_G.sh <arpa> <lang>"
    exit 1;
fi


cat $arpa |
utils/find_arpa_oovs.pl $lang/words.txt  > oovs.txt

arpa2fst $arpa | fstprint | \
utils/remove_oovs.pl oovs.txt | \
utils/eps2disambig.pl | utils/s2eps.pl | fstcompile --isymbols=$lang/words.txt \
   --osymbols=$lang/words.txt  --keep_isymbols=false --keep_osymbols=false | \
fstrmepsilon | fstarcsort --sort_type=ilabel > $lang/G.fst
fstisstochastic $lang/G.fst

rm oovs.txt
