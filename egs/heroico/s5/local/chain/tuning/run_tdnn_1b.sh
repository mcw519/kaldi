#!/usr/bin/env bash

# 1b is as 1a but a re-tuned model with quite a few changes, including moving to
#   a resnet-style factored TDNN-F model.
#
# ./local/chain/compare_wer.sh exp/chain/tdnn1a_sp exp/chain/tdnn1b_sp
# System                  tdnn1a_sp tdnn1b_sp
# %WER        devtest       13.10     12.99
# %WER           test       15.53      9.64
# %WER         native       10.14      6.13
# %WER      nonnative       19.78     12.47
# Final train prob          -0.0233   -0.0442
# Final valid prob          -0.0720   -0.0726
# Final train prob (xent)   -0.8107   -0.9759
# Final valid prob (xent)   -0.9898   -0.9964
# Num-params                 6559440   3318224

# steps/info/chain_dir_info.pl  exp/chain/tdnn1b_sp
# exp/chain/tdnn1b_sp: num-iters=34 nj=2..5 num-params=3.3M dim=40+100->1384 combine=-0.044->-0.044 (over 1) xent:train/valid[21,33,final]=(-1.30,-0.993,-0.976/-1.28,-1.01,-0.996) logprob:train/valid[21,33,final]=(-0.071,-0.050,-0.044/-0.093,-0.076,-0.073)

# Set -e here so that we catch if any executable fails immediately
set -euo pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=0
decode_nj=10
train_set=train
test_sets="native nonnative devtest test"
gmm=tri3b
nnet3_affix=

# The rest are configs specific to this script.  Most of the parameters
# are just hardcoded at this level, in the commands below.
affix=1b   # affix for the TDNN directory name
tree_affix=
train_stage=-10
get_egs_stage=-10
decode_iter=

num_leaves=3500

# training options
# training chunk-options
chunk_width=140,100,160
# we don't need extra left/right context for TDNN systems.
dropout_schedule='0,0@0.20,0.3@0.50,0'
common_egs_dir=
xent_regularize=0.1

# training options
srand=0
remove_egs=true
reporting_email=

#decode options
test_online_decoding=false  # if true, it will run the last decoding stage.

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 11" if you have already
# run those things.
local/nnet3/run_ivector_common.sh --stage $stage \
                                  --train-set $train_set \
                                  --gmm $gmm \
                                  --nnet3-affix "$nnet3_affix" || exit 1;

# Problem: We have removed the "train_" prefix of our training set in
# the alignment directory names! Bad!
gmm_dir=exp/$gmm
ali_dir=exp/${gmm}_ali_${train_set}_sp
tree_dir=exp/chain${nnet3_affix}/tree_sp${tree_affix:+_$tree_affix}
lang=data/lang_chain
lat_dir=exp/chain${nnet3_affix}/${gmm}_${train_set}_sp_lats
dir=exp/chain${nnet3_affix}/tdnn${affix}_sp
train_data_dir=data/${train_set}_sp_hires
lores_train_data_dir=data/${train_set}_sp
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires

for f in $gmm_dir/final.mdl $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp \
    $lores_train_data_dir/feats.scp $ali_dir/ali.1.gz; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done

if [ $stage -le 10 ]; then
  echo "$0: creating lang directory $lang with chain-type topology"
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  if [ -d $lang ]; then
    if [ $lang/L.fst -nt data/lang/L.fst ]; then
      echo "$0: $lang already exists, not overwriting it; continuing"
    else
      echo "$0: $lang already exists and seems to be older than data/lang..."
      echo " ... not sure what to do.  Exiting."
      exit 1;
    fi
  else
    cp -r data/lang $lang
    silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
    nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
    # Use our special topology... note that later on may have to tune this
    # topology.
    steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
  fi
fi

if [ $stage -le 11 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --nj 75 --cmd "$train_cmd" ${lores_train_data_dir} \
    data/lang $gmm_dir $lat_dir
  rm $lat_dir/fsts.*.gz # save space
fi

if [ $stage -le 12 ]; then
  # Build a tree using our new topology.  We know we have alignments for the
  # speed-perturbed data (local/nnet3/run_ivector_common.sh made them), so use
  # those.  The num-leaves is always somewhat less than the num-leaves from
  # the GMM baseline.
   if [ -f $tree_dir/final.mdl ]; then
     echo "$0: $tree_dir/final.mdl already exists, refusing to overwrite it."
     exit 1;
  fi
  steps/nnet3/chain/build_tree.sh \
    --cmd "$train_cmd" \
    --frame-subsampling-factor 3 \
    --context-opts "--context-width=2 --central-position=1" \
    $num_leaves \
    ${lores_train_data_dir} \
    $lang $ali_dir $tree_dir
fi


if [ $stage -le 13 ]; then
  mkdir -p $dir
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)
  affine_opts="l2-regularize=0.03 dropout-proportion=0.0 dropout-per-dim-continuous=true"
  tdnnf_opts="l2-regularize=0.03 dropout-proportion=0.0 bypass-scale=0.66"
  linear_opts="l2-regularize=0.03 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.03"
  output_opts="l2-regularize=0.015"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-dropout-layer name=tdnn1 $affine_opts dim=768
  tdnnf-layer name=tdnnf2 $tdnnf_opts dim=768 bottleneck-dim=64 time-stride=1
  tdnnf-layer name=tdnnf3 $tdnnf_opts dim=768 bottleneck-dim=64 time-stride=1
  tdnnf-layer name=tdnnf4 $tdnnf_opts dim=768 bottleneck-dim=64 time-stride=1
  tdnnf-layer name=tdnnf5 $tdnnf_opts dim=768 bottleneck-dim=64 time-stride=0
  tdnnf-layer name=tdnnf6 $tdnnf_opts dim=768 bottleneck-dim=64 time-stride=3
  tdnnf-layer name=tdnnf7 $tdnnf_opts dim=768 bottleneck-dim=64 time-stride=3
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=768 bottleneck-dim=64 time-stride=3
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=768 bottleneck-dim=64 time-stride=3
  tdnnf-layer name=tdnnf10 $tdnnf_opts dim=768 bottleneck-dim=64 time-stride=3
  tdnnf-layer name=tdnnf11 $tdnnf_opts dim=768 bottleneck-dim=64 time-stride=3
  linear-component name=prefinal-l dim=192 $linear_opts

  ## adding the layers for chain branch
  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts big-dim=768 small-dim=192
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  # adding the layers for xent branch
  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts big-dim=768 small-dim=192
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi


if [ $stage -le 14 ]; then
  steps/nnet3/chain/train.py \
    --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.online-ivector-dir=$train_ivector_dir \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient=0.1 \
    --chain.l2-regularize=0.0 \
    --chain.apply-deriv-weights=false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=8 \
    --trainer.frames-per-iter=3000000 \
    --trainer.optimization.num-jobs-initial=2 \
    --trainer.optimization.num-jobs-final=5 \
    --trainer.optimization.initial-effective-lrate=0.001 \
    --trainer.optimization.final-effective-lrate=0.0001 \
    --trainer.num-chunk-per-minibatch=128,64 \
    --egs.chunk-width=$chunk_width \
    --egs.dir="$common_egs_dir" \
    --egs.opts="--frames-overlap-per-eg 0" \
    --cleanup.remove-egs=$remove_egs \
    --use-gpu=true \
    --reporting.email="$reporting_email" \
    --feat-dir=$train_data_dir \
    --tree-dir=$tree_dir \
    --lat-dir=$lat_dir \
    --dir=$dir  || exit 1;
fi

if [ $stage -le 15 ]; then
  # Note: it's not important to give mkgraph.sh the lang directory with the
  # matched topology (since it gets the topology file from the model).
  utils/mkgraph.sh \
    --self-loop-scale 1.0 \
    data/lang_test \
    $tree_dir \
    $tree_dir/graph || exit 1;
fi

if [ $stage -le 16 ]; then
  frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    (
    nspk=$(wc -l <data/${data}_hires/spk2utt)
    steps/nnet3/decode.sh \
      --acwt 1.0 \
      --post-decode-acwt 10.0 \
      --frames-per-chunk $frames_per_chunk \
      --nj $nspk \
      --cmd "$decode_cmd" \
      --num-threads 4 \
      --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${data}_hires \
      $tree_dir/graph \
      data/${data}_hires \
      ${dir}/decode_${data} || exit 1;
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

# Not testing the 'looped' decoding separately, because for
# TDNN systems it would give exactly the same results as the
# normal decoding.

if $test_online_decoding && [ $stage -le 17 ]; then
  # note: if the features change (e.g. you add pitch features), you will have to
  # change the options of the following command line.
  steps/online/nnet3/prepare_online_decoding.sh \
    --mfcc-config conf/mfcc_hires.conf \
    $lang \
    exp/nnet3${nnet3_affix}/extractor \
    ${dir} \
    ${dir}_online

  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    (
    nspk=$(wc -l <data/${data}_hires/spk2utt)
    # note: we just give it "data/${data}" as it only uses the wav.scp, the
    # feature type does not matter.
    steps/online/nnet3/decode.sh \
      --acwt 1.0 --post-decode-acwt 10.0 \
      --nj $nspk --cmd "$decode_cmd" \
      $tree_dir/graph data/${data} ${dir}_online/decode_${data} || exit 1
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

exit 0;

# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# End:
