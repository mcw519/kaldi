#!/bin/bash

# Copyright Author: Meng Wu

cmd=queue.pl
id_prefix=
id_suffix=

. utils/parse_options.sh

if [ $# != 2 ]; then
    echo "Usage: $0 <lat-dir> <out-dir>"
    echo " --cmd <run.pl|queue.pl>,	# default queue.pl"
    echo " --id-prefix=<prefix>		# Prefix for utterance id, default empty"
    echo " --id-suffix=<suffix>		# Suffix for utterance id, default empty"
    exit 1;
fi

lat_dir=$1
dir=$2

export LC_ALL=C
set -e;
mkdir -p $dir/temp/log
num_jobs=$(cat $lat_dir/num_jobs) || exit 1;

echo "$0 :copy old lattice"
$cmd JOB=1:$num_jobs $dir/temp/log/copy_lattice_ori.JOB.log \
	lattice-copy "ark:gunzip -c $lat_dir/lat.JOB.gz |" ark,scp:$dir/temp/lats.JOB.ark,$dir/temp/lats.JOB.scp \
	|| exit 1;

echo "$0 :change by prefix|suffix"
if [[ ! -z $id_prefix || ! -z $id_suffix ]]; then
	for i in $(seq 1 $(cat $lat_dir/num_jobs)); do
		cat $dir/temp/lats.${i}.scp | sed -e "s/ /${id_suffix} /g" | sed -e "s/^/${id_prefix}/g" > $dir/temp/combined_lats.${i}.scp
		sort -u $dir/temp/combined_lats.${i}.scp > $dir/temp/combined_lats_sorted.${i}.scp
	done
fi

echo "$0 :copy lattice to $dir"
$cmd JOB=1:$num_jobs $dir/temp/log/copy_lattice_new.JOB.log \
	lattice-copy scp:$dir/temp/combined_lats_sorted.JOB.scp "ark:|gzip -c > $dir/lat.JOB.gz" \
	|| exit 1;


for f in num_jobs cmvn_opts final.mdl tree; do
	cp $lat_dir/$f $dir/$f
done


rm -r $dir/temp #save space.
