#!/bin/bash

if [ $# -ne 1 ]; then
   echo "Prepared data folder which contain download Subtitle."
   echo "Usage: $0 <data-dir>"
   exit 1;
fi

src=src
data=$1

if [ ! -d $data/subtitle-zh-TW ]; then
  echo "$0: missing subtitle-zh-TW dir in $data folder"
  exit 1;
fi

num_wav=`wc -l $data/wav.scp`
num_subtitles=`ls $data/subtitle-zh-TW/*.vtt | wc -l`

if [ $num_subtitles == 0 ]; then
  echo "$0: no subtitles can be used, STOP"
  exit 1;
else
  echo "$0: only have ${num_subtitles}/${num_wav} has subtitles could be used"
fi

for i in $data/subtitle-zh-TW/*.vtt; do
  python $src/srt_transform.py $i
done

cat $data/subtitle-zh-TW/*.vtt.segments | sort > $data/segments
cat $data/subtitle-zh-TW/*.vtt.text | sort > $data/text
awk '{print $1,$2}' $data/segments > $data/utt2spk
utt2spk_to_spk2utt.pl $data/utt2spk > $data/spk2utt

./utils/fix_data_dir.sh $data
