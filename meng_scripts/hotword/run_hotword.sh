#!/bin/bash

# Copyright 2020 (author: Meng Wu)

weight=0.9 # weight for each hotword

. parse_options.sh || exit 1;

if [ $# != 5 ]; then
  cat >&2 <<EOF
Usage: $0 [options] <words.txt> <output dir> <hotword table> <unigram> <old HCLG.fst>
Note:
  Format(x): <type id> <content>
    type 1:
      Chinese word, "每日一物", "吳孟哲"
    type 2:
      English word, "LEONA"
    type 3:
      Customize word pair, first column sould be your customize word and the others are how to spell.
      "TAYLOR-SWIFT TAYLOR SWIFT", "LMFAO L M F A O"
Options:
  --weight <weight> # weight for each hotword, default=0.9
EOF
   exit 1;
fi

wd_table=$1
outputdir=$2
hotword=$3
unigram=$4
HCLG=$5

new_wd_table=$outputdir/words.txt
C=$outputdir/C.txt.fst
Cfst=$outputdir/C.fst

python3 hotword_context.py $hotword $wd_table $unigram $weight $outputdir

fstcompile --isymbols=$new_wd_table --osymbols=$new_wd_table $C | fstarcsort --sort_type=ilabel > $Cfst
fstcompose $HCLG $Cfst $outputdir/HCLG.fst
