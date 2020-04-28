#!/bin/bash


. parse_options.sh || exit 1;

if [ $# != 5 ]; then
  cat >&2 <<EOF
Usage: $0 [options] <words.txt> <output dir> <hotword table> <unigram> <old HCLG.fst>
Note:
Options:
EOF
   exit 1;
fi

wd_table=$1
outputdir=$2
hotword=$3
unigram=$4
HCLG=$5

weight=0.9
new_wd_table=$outputdir/words.txt
C=$outputdir/C.txt.fst
Cfst=$outputdir/C.fst

python hotword_context.py $hotword $wd_table $unigram $weight $outputdir

fstcompile --isymbols=$new_wd_table --osymbols=$new_wd_table $C | fstarcsort --sort_type=ilabel > $Cfst
fstcompose $HCLG $Cfst $outputdir/HCLG.fst
