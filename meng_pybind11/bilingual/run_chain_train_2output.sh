#!/bin/bash

# Copyright 2019 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
#           2020 Meng Wu
# Apache 2.0

set -e

train_script="train_bilingual.py" # others "train_bilingual_lid_discriminator.py"

nj=10

train_1_data_dir=/nfs/TPAICSASR03/meng/multi_lang/data/mandarin-train
train_1_ivector_dir=/nfs/TPAICSASR03/meng/multi_lang/data/mandarin-train/ivector
lat_1_dir=/nfs/TPAICSASR03/meng/multi_lang/exp/tri3_multi/alignemnt/mandarin_train
train_2_data_dir=/nfs/TPAICSASR03/meng/multi_lang/data/LS360
train_2_ivector_dir=/nfs/TPAICSASR03/meng/multi_lang/data/LS360/ivector
lat_2_dir=/nfs/TPAICSASR03/meng/multi_lang/exp/tri3_multi/alignemnt/ls360

chain_dir=/nfs/TPAICSASR03/meng/multi_lang/exp/mandarin_ls360_train_2output
tree_dir=/nfs/TPAICSASR03/meng/multi_lang/exp/mandarin_ls360_train/tree_monophone.1000

train_ivector=true
online_cmvn=true
num_epochs=5
dropout_schedule=0,0@0.20,0.5@0.50,0      # you might set this to 0,0 or 0.5,0.5 to train.
frame_subsampling_factor=3
feat_type="delta"
lang=default

# You should know how to calculate your model's left/right context **manually**
model_left_context=28
model_right_context=28
egs_left_context=$(($model_left_context + 1))
egs_right_context=$(($model_right_context + 1))
frames_per_eg=150,110,90
frames_per_iter=150000 #1500000
minibatch_size=128

hidden_dim=1024
bottleneck_dim=128
prefinal_bottleneck_dim=256
kernel_size_list="3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3" # comma separated list
subsampling_factor_list="1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1" # comma separated list

log_level=info # valid values: debug, info, warning

# true to save network output as kaldi::CompressedMatrix
# false to save it as kaldi::Matrix<float>
save_nn_output_as_compressed=false

num_workers=4


. ./path.sh
. ./cmd.sh

. parse_options.sh


DoPhoneLMandFST=0
DumpRawEgs=0
ProcessEgs=0
CombineEgs=0
MergeEgs=0
AlignEgNumber=0
Training=1



if  [ $DoPhoneLMandFST -eq 1 ]; then
  echo "$0: Making Phone LM and denominator and normalization FST"
  mkdir -p $chain_dir/den_fsts/log

  # We may later reorganize this.
  # We using same tree for different language
  cp $tree_dir/tree $chain_dir/${lang}.tree

  echo "$0: creating phone language-model"
  $train_cmd $chain_dir/den_fsts/log/make_phone_lm_${lang}.log \
    chain-est-phone-lm --num-extra-lm-states=2000 \
       "ark:gunzip -c $tree_dir/ali.*.gz | ali-to-phones $tree_dir/final.mdl ark:- ark:- |" \
       $chain_dir/den_fsts/${lang}.phone_lm.fst
  mkdir -p $chain_dir/init
  copy-transition-model $tree_dir/final.mdl $chain_dir/init/${lang}_trans.mdl
  echo "$0: creating denominator FST"
  $train_cmd $chain_dir/den_fsts/log/make_den_fst.log \
     chain-make-den-fst $chain_dir/${lang}.tree $chain_dir/init/${lang}_trans.mdl $chain_dir/den_fsts/${lang}.phone_lm.fst \
     $chain_dir/den_fsts/${lang}.den.fst $chain_dir/den_fsts/${lang}.normalization.fst || exit 1;
  
  for i in 1 2; do
    cp $chain_dir/${lang}.tree $chain_dir/${lang}_${i}.tree
    cp $chain_dir/init/${lang}_trans.mdl $chain_dir/init/${lang}_${i}_trans.mdl
    cp $chain_dir/den_fsts/${lang}.normalization.fst $chain_dir/den_fsts/${lang}_${i}.normalization.fst
    cp $chain_dir/den_fsts/${lang}.den.fst $chain_dir/den_fsts/${lang}_${i}.den.fst
    cp $chain_dir/den_fsts/${lang}.phone_lm.fst $chain_dir/den_fsts/${lang}_${i}.phone_lm.fst
  
  done
fi


if [ $DumpRawEgs -eq 1 ]; then
  echo "$0: about to dump raw egs."
  # Dump raw egs.
  steps/chain2/get_raw_egs.sh --cmd "$train_cmd" \
    --lang "${lang}_1" \
    --online-cmvn $online_cmvn \
    --online-ivector-dir "$train_1_ivector_dir" \
    --left-context $egs_left_context \
    --right-context $egs_right_context \
    --frame-subsampling-factor $frame_subsampling_factor \
    --alignment-subsampling-factor $frame_subsampling_factor \
    --frames-per-chunk $frames_per_eg \
    --feat-type $feat_type \
    ${train_1_data_dir} ${chain_dir} ${lat_1_dir} ${chain_dir}/raw_egs_1

  steps/chain2/get_raw_egs.sh --cmd "$train_cmd" \
    --lang "${lang}_2" \
    --online-cmvn $online_cmvn \
    --online-ivector-dir "$train_2_ivector_dir" \
    --left-context $egs_left_context \
    --right-context $egs_right_context \
    --frame-subsampling-factor $frame_subsampling_factor \
    --alignment-subsampling-factor $frame_subsampling_factor \
    --frames-per-chunk $frames_per_eg \
    --feat-type $feat_type \
    ${train_2_data_dir} ${chain_dir} ${lat_2_dir} ${chain_dir}/raw_egs_2
fi

if [ $ProcessEgs -eq 1 ]; then
  echo "$0: about to process egs"
  steps/chain2/process_egs.sh  --cmd "$train_cmd" \
      --num-repeats 1 \
    ${chain_dir}/raw_egs_1 ${chain_dir}/processed_egs_1

  steps/chain2/process_egs.sh  --cmd "$train_cmd" \
      --num-repeats 1 \
    ${chain_dir}/raw_egs_2 ${chain_dir}/processed_egs_2
fi


if [ $CombineEgs -eq 1 ]; then
  echo "$0: combind egs"
  ./chain_bilingual/combine_egs.sh 2 ${chain_dir}/processed_egs_1 ${chain_dir}/processed_egs_2 ${chain_dir}/egs_combine_from_process
fi

info_file=$chain_dir/raw_egs_1/info.txt
feat_dim=$(grep 'feat_dim' $info_file | awk '{print $NF}')
ivector_dim=0
ivector_period=0
if $train_ivector; then
  ivector_dim=$(grep 'ivector_dim' $info_file | awk '{print $NF}')
  ivector_period=$(cat $train_1_ivector_dir/ivector_period)
fi
echo "ivector_dim: $ivector_dim", "ivector_period, $ivector_period"

merged_egs_dir=merged_egs_chain
if [ $MergeEgs -eq 1 ]; then
  echo "$0: merging egs"

  mkdir -p $chain_dir/${merged_egs_dir}
  num_egs=$(ls -1 $chain_dir/egs_combine_from_process/train.*.scp | wc -l)

  $train_cmd --max-jobs-run $nj JOB=1:$num_egs $chain_dir/${merged_egs_dir}/log/merge_egs.JOB.log \
    nnet3-chain-copy-egs --weights=ark:$chain_dir/egs_combine_from_process/train.weight.JOB.ark scp:$chain_dir/egs_combine_from_process/train.JOB.scp ark:- \| \
    nnet3-chain-shuffle-egs ark:- ark:- \| \
    nnet3-chain-merge-egs --multilingual-eg=true --minibatch-size=$minibatch_size ark:- \
      ark,scp:$chain_dir/${merged_egs_dir}/cegs.JOB.ark,$chain_dir/${merged_egs_dir}/cegs.JOB.scp || exit 1

  #rm -f $chain_dir/raw_egs_1/cegs.*.ark
  mkdir ${merged_egs_dir}_valid
  nnet3-chain-copy-egs --weights=ark:$chain_dir/egs_combine_from_process/train_subset.weight.ark scp:$chain_dir/egs_combine_from_process/train_subset.scp ark:- \| \
  nnet3-chain-shuffle-egs ark:- ark:- \| \
  nnet3-chain-merge-egs --multilingual-eg=true --minibatch-size=$minibatch_size ark:- \
    ark,scp:$chain_dir/${merged_egs_dir}_valid/cegs.ark,$chain_dir/${merged_egs_dir}_valid/cegs.scp || exit 1
fi

training_eg_dir=egs_chain2_for_training
# we have to make sure each scp file holding the same number of lines,
# as we will load them with multiple workers in PyTorch and there is an
# assumption in DDP training that num-mininbatches should be equal
# across workers.
if [ $AlignEgNumber -eq 1 ]; then
  echo "$0: align eg numbers in each scp file"

  mkdir -p $chain_dir/${training_eg_dir}/tmp_scp_dir
  steps/chain2/align_eg_numbers.sh $chain_dir/${merged_egs_dir} $chain_dir/${training_eg_dir}/tmp_scp_dir

  # TODO: make this more efficient as for each ark file there are only few arks
  #       we really need to copy from other ark files.
  num_egs=$(ls -1 $chain_dir/${training_eg_dir}/tmp_scp_dir/*.scp | wc -l)
  $train_cmd --max-jobs-run $nj JOB=1:$num_egs $chain_dir/${training_eg_dir}/log/copy_egs.JOB.log \
    nnet3-chain-copy-egs scp:$chain_dir/${training_eg_dir}/tmp_scp_dir/cegs.JOB.scp \
      ark,scp:$chain_dir/${training_eg_dir}/cegs.JOB.ark,$chain_dir/${training_eg_dir}/cegs.JOB.scp || exit 1

  #rm -r $chain_dir/${training_eg_dir}/tmp_scp_dir
  #rm -f $chain_dir/${merged_egs_dir}/cegs.*.ark

fi

output_dim=$(grep 'num_leaves' $info_file | awk '{print $NF}')
train_dir=train${train_affix}
if [ $Training -eq 1 ]; then
  echo "$0: training..."

  mkdir -p $chain_dir/$train_dir/tensorboard
  train_checkpoint=
  if [[ -f $chain_dir/$train_dir/best_model.pt ]]; then
    train_checkpoint=$chain_dir/$train_dir/best_model.pt
  fi

  INIT_FILE=$chain_dir/$train_dir/ddp_init
  rm -f $INIT_FILE # delete old one before starting
  init_method=file://$(readlink -f $INIT_FILE)
  # use '127.0.0.1' for training on a single machine
  init_method=tcp://127.0.0.1:7275
  echo "$0: init method is $init_method"

  num_epochs=$num_epochs
  lr=1e-3
  
  # use_ddp = false & world_size = 1: training model with one GPU
  # use_ddp = true & use_multiple_machine = false: training model with multiple GPUs on a single machine
  # use_ddp = true & use_multiple_machine = true:  training model with GPU on multiple machines

  use_ddp=true
  world_size=$num_workers
  use_multiple_machine=false
  # you can assign GPUs with --device-ids "$device_ids"
  # device_ids="4, 5, 6, 7"
  if $use_multiple_machine ; then
    # suppose you are using Sun GridEngine
    cuda_train_cmd="$cuda_train_cmd --gpu 1 JOB=1:$world_size $chain_dir/$train_dir/logs/job.JOB.log"
  else
    cuda_train_cmd="$cuda_train_cmd --gpu $world_size $chain_dir/$train_dir/logs/train.log"
  fi
  
  $cuda_train_cmd python3 ./chain_bilingual/$train_script \
        --bottleneck-dim $bottleneck_dim \
        --checkpoint=${train_checkpoint:-} \
        --dir $chain_dir/$train_dir \
        --feat-dim $feat_dim \
        --hidden-dim $hidden_dim \
        --is-training true \
        --ivector-dim $ivector_dim \
        --kernel-size-list "$kernel_size_list" \
        --log-level $log_level \
        --output-dim $output_dim \
        --prefinal-bottleneck-dim $prefinal_bottleneck_dim \
        --subsampling-factor-list "$subsampling_factor_list" \
        --train.cegs-dir $chain_dir/$training_eg_dir \
        --train.ddp.init-method $init_method \
        --train.ddp.multiple-machine $use_multiple_machine \
        --train.ddp.world-size $world_size \
        --train.den-fst $chain_dir/den_fsts/${lang}.den.fst \
        --train.dropout-schedule "$dropout_schedule" \
        --train.egs-left-context $egs_left_context \
        --train.egs-right-context $egs_right_context \
        --train.l2-regularize 5e-5 \
        --train.leaky-hmm-coefficient 0.1 \
        --train.lr $lr \
        --train.num-epochs $num_epochs \
        --train.use-ddp $use_ddp \
        --train.valid-cegs-scp $chain_dir/${merged_egs_dir}_valid/cegs.scp \
        --train.xent-regularize 0.1 || exit 1;
fi

