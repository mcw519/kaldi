#!/bin/bash

segment_file=$1
wav_scp=$2
output_dir=$3

if [[ $# != 3 ]]; then
	echo " $0 <segment-file> <wav.scp> <out-dir>"
	exit 1;
fi

mkdir -p $output_dir

while read file; do
	#echo $file
	utt_id=`echo $file | cut -d " " -f 1 `;
	#echo $utt_id
	wav_id=`echo $file | cut -d " " -f 2 `;
	start_time=`echo $file | cut -d " " -f 3 `
	end_time=`echo $file | cut -d " " -f 4 `
	wav_path=`grep "^$wav_id " $wav_scp | cut -f 2 -d' '`;
	echo "ffmpeg -i $wav_path -ss $start_time -to $end_time -v 0 -c copy $output_dir/${utt_id}.wav" >> exec_file
done < $1

chmod +x exec_file
./exec_file
rm exec_file
