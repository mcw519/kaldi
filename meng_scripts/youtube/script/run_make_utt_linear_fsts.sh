#!/bin/bash

cmd=run.pl
nj=40

if [ $# != 2 ]; then
  echo "Usage: $0 <data-dir> <model-dir>"
  exit 1;
fi

data=`make_absolute.sh $1`
model=$2

split_data.sh $data $nj

$cmd JOB=1:$nj $data/log/make_utt_graph.JOB \
  ./make_utt_linear_fsts.sh $data/split$nj/JOB $model

cp -r $data/split$nj/1/graph $data/graph
rm $data/graph/HCLG*
cat $data/split$nj/*/graph/HCLG.fsts.scp | sort > $data/graph/HCLG.fsts.scp
