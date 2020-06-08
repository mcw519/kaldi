#!/bin/bash

nj=1
device_id=0
# You should know how to calculate your model's left/right context **manually**
model_left_context=28
model_right_context=28
egs_left_context=$[$model_left_context + 1]
egs_right_context=$[$model_right_context + 1]
frames_per_eg=150,110,90
frames_per_iter=1500000
minibatch_size=128
ivector_dim=100
ivector_period=10
hidden_dim=1024
bottleneck_dim=128
prefinal_bottleneck_dim=256
kernel_size_list="3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3" # comma separated list
subsampling_factor_list="1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1" # comma separated list

log_level=info # valid values: debug, info, warning
save_nn_output_as_compressed=false

feat_type="delta"
online_cmvn=true
lang="default"


. utils/parse_options.sh
. cmd.sh

if [ $# != 3 ]; then
    echo "Usage: $0 <graph-dir> <data-dir> <chain-dir>"
    echo " --cmd <run.pl|queue.pl>,     # default queue.pl"
    exit 1;
fi

graphdir=$1
evaldir=$2
chaindir=$3
evalname=`basename $evaldir`
ivector_scp=$evaldir/ivector/ivector_online.scp


Inference=1
Decode=1

if [ $Inference -eq 1 ]; then
  echo "$0: inference: computing likelihood"
  mkdir -p $chaindir/inference/$evalname
  best_epoch=$(cat $chaindir/train/best-epoch-info | grep 'best epoch' | awk '{print $NF}')
  inference_checkpoint=$chaindir/train/epoch-${best_epoch}.pt
  feat_dim=43 #$(cat $chaindir/egs/info/feat_dim)
  output_dim=288 #$(cat $chaindir/egs/info/num_pdfs)
  feat_scp="$evaldir/feats.scp"
  if $online_cmvn; then
    if [[ "$feat_type" == "delta" ]]; then
      apply-cmvn-online --spk2utt=ark:$evaldir/spk2utt $chaindir/raw_egs/global_cmvn.stats \
          scp:$evaldir/feats.scp ark:- | add-deltas --print-args=false --delta-order=2 --delta-window=2 \
          ark:- ark,scp:$evaldir/online_cmvn_feats.ark,$evaldir/online_cmvn_feats.scp
    fi
    feat_scp="$evaldir/online_cmvn_feats.scp"
  fi


  run.pl --gpu 1 $chaindir/inference/logs/$evalname.log \
    python3 ./chain/inference.py \
      --bottleneck-dim $bottleneck_dim \
      --checkpoint $inference_checkpoint \
      --dir $chaindir/inference/$evalname \
      --feat-dim $feat_dim \
      --feats-scp $feat_scp \
      --ivector-dim $ivector_dim \
      --ivector-period $ivector_period \
      --ivector-scp "$ivector_scp" \
      --hidden-dim $hidden_dim \
      --is-training false \
      --kernel-size-list "$kernel_size_list" \
      --log-level $log_level \
      --model-left-context $model_left_context \
      --model-right-context $model_right_context \
      --output-dim $output_dim \
      --prefinal-bottleneck-dim $prefinal_bottleneck_dim \
      --save-as-compressed $save_nn_output_as_compressed \
      --subsampling-factor-list "$subsampling_factor_list" || exit 1
fi

if [ $Decode -eq 1 ]; then
  echo "$0: decoding"
  ./local/decode.sh \
    --nj $nj \
    $graphdir \
    $chaindir/init/${lang}_trans.mdl \
    $chaindir/inference/$evalname/nnet_output.scp \
    $chaindir/decode_res/$evalname
fi

