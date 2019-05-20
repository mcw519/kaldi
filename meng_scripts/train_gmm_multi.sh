#!/bin/bash

train_all=$1 
lang=$2
exp=$3

train_cmd=queue.pl
numjobs=400

meng_scripts/train_mono_multi.sh --boost-silence 1.25 --cmd $train_cmd --nj $numjobs ${train_all}.100k $lang $exp/mono || exit 1;
meng_scripts/align_si_multi.sh --boost-silence 1.25 --cmd $train_cmd --nj $numjobs $train_all $lang $exp/mono $exp/mono_ali

meng_scripts/train_deltas_multi.sh --boost-silence 1.25 --cmd $train_cmd 2500 20000 $train_all $lang $exp/mono_ali $exp/tri1
meng_scripts/align_si_multi.sh --cmd $train_cmd --nj $numjobs $train_all $lang $exp/tri1 $exp/tri1_ali

meng_scripts/train_lda_mllt_multi.sh --cmd $train_cmd 2500 20000 $train_all $lang $exp/tri1_ali $exp/tri2
meng_scripts/align_si_multi.sh --cmd $train_cmd --nj $numjobs $train_all $lang $exp/tri2 $exp/tri2_ali

meng_scripts/train_sat_multi.sh --cmd $train_cmd 3500 100000 $train_all $lang $exp/tri2_ali $exp/tri3
meng_scripts/align_fmllr_lats_multi.sh --cmd $train_cmd --nj $numjobs $train_all $lang $exp/tri3 $exp/tri3_ali
