#!/bin/bash

# Copyright 2020 Meng Wu

data_dir=$1
stage=-3

if [ $# != 1 ]; then
  echo "Usage: $0 <download-dir>"
  exit 1;
fi


if [ $stage -le -4 ]; then
  mkdir -p $data_dir/m4a
  mv $data_dir/*.m4a $data_dir/m4a
  mkdir $data_dir/subtitle-zh-TW
  mkdir $data_dir/subtitle-en
  mv $data_dir/*zh-TW.vtt $data_dir/subtitle-zh-TW
  mv $data_dir/*en.vtt $data_dir/subtitle-en
  rm $data_dir/*.vtt
fi


if [ $stage -le -3 ]; then
  ## check filename not include "-" and rename to "dash".
  ls $data_dir/m4a | cut -f 1 -d "." > $data_dir/list_wav
  ls $data_dir/subtitle-zh-TW | cut -f 1 -d "." > $data_dir/list_md_text
  ls $data_dir/subtitle-en | cut -f 1 -d "." > $data_dir/list_en_text
  while read file; do
    if [[ $file =~ "-" ]]; then
      new_name=`echo $file | sed 's/-/dash/g'`
      mv -- $data_dir/m4a/${file}.m4a $data_dir/m4a/${new_name}.m4a
    else
      continue
    fi
  done < $data_dir/list_wav

  while read file; do
    if [[ $file =~ "-" ]]; then
      new_name=`echo $file | sed 's/-/dash/g'`
      mv -- $data_dir/subtitle-zh-TW/${file}.zh-TW.vtt $data_dir/subtitle-zh-TW/${new_name}.zh-TW.vtt
    else
      continue
    fi
  done < $data_dir/list_md_text
  while read file; do
    if [[ $file =~ "-" ]]; then
      new_name=`echo $file | sed 's/-/dash/g'`
      mv -- $data_dir/subtitle-en/${file}.en.vtt $data_dir/subtitle-en/${new_name}.en.vtt
    else
      continue
    fi
  done < $data_dir/list_en_text

fi


if [ $stage -le -2 ]; then
  ## ffmpeg re-sampling and change wav format.
  mkdir -p $data_dir/wav-16k
  ls $data_dir/m4a | cut -f 1 -d "." > $data_dir/list_wav

  for i in $(cat $data_dir/list_wav);do
    ffmpeg -i $data_dir/m4a/$i.m4a -ar 16000 -ac 1 $data_dir/wav-16k/$i.wav
  done

fi

rm $data_dir/list_*_text $data_dir/list_wav

if [ $stage -le -1 ]; then
  ## create kaldi fotmat
  mkdir -p $data_dir/kaldi.format
  ls $data_dir/wav-16k | sed 's/.wav//g' | sort > $data_dir/kaldi.format/id
  find $data_dir/wav-16k -name *.wav | sort > $data_dir/kaldi.format/path

  paste $data_dir/kaldi.format/id $data_dir/kaldi.format/id | sed 's/\t/ /g' | sort > $data_dir/kaldi.format/utt2spk
  paste $data_dir/kaldi.format/id $data_dir/kaldi.format/id | sed 's/\t/ /g' | sort > $data_dir/kaldi.format/spk2utt
  paste $data_dir/kaldi.format/id $data_dir/kaldi.format/path | sed 's/\t/ /g' | sort > $data_dir/kaldi.format/wav.scp

fi
