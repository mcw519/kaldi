#!/bin/bash

# Copyright 2020 Meng Wu

if [ $# -ne 2 ]; then
   echo "Usage: $0 <old-data-dir> <confidence-island-dir>"
   exit 1;
fi

data=`make_absolute.sh $1`
data_cleanup=$data.cleanup
conf_dir=$2
cmd=run.pl
nj=20

utils/copy_data_dir.sh $data $data_cleanup
rm $data_cleanup/feats.scp $data_cleanup/cmvn.scp
mkdir -p $data_cleanup/audio

path=`head -n1 $data_cleanup/wav.scp | cut -f 2- -d " "`
base=`basename $path`
wav_path=`echo $path | sed "s/$base//g"`
#echo $wav_path
awk -v wav_path="$wav_path" -v audio_path="$data_cleanup/audio/" -F" " '{print "ffmpeg -i " wav_path "/" $2".wav -ss "$3" -to "$4" -v 0 -c copy "audio_path$1".wav"}' $data/segments > $data_cleanup/exec_segment

echo "$0: generate segmented audio"
a=$data_cleanup/exec_segment
b=''
for i in $(seq 1 $nj); do  b+=" $a.$i"; done

split_scp.pl $data_cleanup/exec_segment $b
chmod +x $data_cleanup/exec_segment.*

$cmd JOB=1:$nj $data_cleanup/log/segment_audio.JOB \
  sh $data_cleanup/exec_segment.JOB

echo "$0: generate new wav.scp"
ls $data_cleanup/audio > $data_cleanup/id
for i in $(cat $data_cleanup/id); do
  a="$i $data_cleanup/audio/$i"
  echo $a | sed "s/\.wav / /g" >> $data_cleanup/wav.scp
done 

cp $conf_dir/segments $data_cleanup/segments
cp $conf_dir/island_text $data_cleanup/text

awk '{print $1,$2}' $data_cleanup/segments > $data_cleanup/utt2spk
utt2spk_to_spk2utt.pl $data_cleanup/utt2spk > $data_cleanup/spk2utt

utils/fix_data_dir.sh $data_cleanup
rm $data_cleanup/exec*

echo "$0: generate arpa ngram file"
cut -f 2- -d " " $data_cleanup/text > $data_cleanup/trans
### TODO
