#!/bin/bash

# Copyright 2014  Guoguo Chen
# Copyright 2020  Meng Wu
# Apache 2.0


tscale=1.0      # transition scale.
loopscale=1.0 #0.1   # scale for self-loops.
lang=
src_folder=src
oov="<SPOKEN_NOISE>"

if [ $# -ne 2 ]; then
   echo "Usage: $0 <data-dir> <model-dir>"
   echo "      needs: <data-dr>/text"
   exit 1;
fi

data=$1
text=$data/text
model_dir=$2

oov=`cat $lang/oov.int`
oov_txt=`cat $lang/oov.txt`

N=`tree-info --print-args=false $model_dir/tree |\
  grep "context-width" | awk '{print $NF}'`
P=`tree-info --print-args=false $model_dir/tree |\
  grep "central-position" | awk '{print $NF}'`

mkdir -p $data/sub_graphs
# utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt | \
cat $text | \
  while read line; do
    uttid=`echo $line | cut -f 1 -d " "`
    wdir=$data/sub_graphs/$uttid
    mkdir -p $wdir
    echo $line | python $src_folder/create_linear_fst.py | fstcompile -isymbols=$lang/words.txt -osymbols=$lang/words.txt |\
      fstarcsort --sort_type=ilabel > $wdir/G.fst 
    fsttablecompose $lang/L_disambig.fst $wdir/G.fst |\
    fstdeterminizestar --use-log=true | fstminimizeencoded |\
    fstarcsort --sort_type=ilabel > $wdir/LG.fst || exit 1;
    fstisstochastic $wdir/LG.fst || echo "$0: $uttid/LG.fst not stochastic."

    # Builds CLG.fst
    clg=$wdir/CLG_${N}_${P}.fst
    fstcomposecontext --context-size=$N --central-position=$P \
      --read-disambig-syms=$lang/phones/disambig.int \
      --write-disambig-syms=$wdir/disambig_ilabels_${N}_${P}.int \
      $wdir/ilabels_${N}_${P} < $wdir/LG.fst | fstdeterminize > $wdir/CLG.fst
    fstisstochastic $wdir/CLG.fst  || echo "$0: $uttid/CLG.fst not stochastic."

    make-h-transducer --disambig-syms-out=$wdir/disambig_tid.int \
      --transition-scale=$tscale $wdir/ilabels_${N}_${P} \
      $model_dir/tree $model_dir/final.mdl > $wdir/Ha.fst

    # Builds HCLGa.fst
    fsttablecompose $wdir/Ha.fst $wdir/CLG.fst | \
      fstdeterminizestar --use-log=true | \
      fstrmsymbols $wdir/disambig_tid.int | fstrmepslocal | \
      fstminimizeencoded > $wdir/HCLGa.fst
    fstisstochastic $wdir/HCLGa.fst ||\
      echo "$0: $uttid/HCLGa.fst is not stochastic"

    add-self-loops --self-loop-scale=$loopscale --reorder=true \
      $model_dir/final.mdl < $wdir/HCLGa.fst > $wdir/HCLG.fst

    if [ $tscale == 1.0 -a $loopscale == 1.0 ]; then
      fstisstochastic $wdir/HCLG.fst ||\
        echo "$0: $uttid/HCLG.fst is not stochastic."
    fi

    echo "$uttid $wdir/HCLG.fst" >> $data/sub_graphs/HCLG.fsts.scp
  done

mkdir -p $data/graph
cp -r $lang/* $data/graph

am-info --print-args=false $model_dir/final.mdl |\
 grep pdfs | awk '{print $NF}' > $data/graph/num_pdfs

# Creates the graph table.
fstcopy scp:$data/sub_graphs/HCLG.fsts.scp \
  "ark,scp:$data/graph/HCLG.fsts,$data/graph/HCLG.fsts.scp"
