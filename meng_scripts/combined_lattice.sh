#!/bin//bash

# Copyright Author: Meng Wu

cmd=queue.pl #run.pl

if [[ $# -lt 3 ]]; then
	echo " $0 <data> <des-dir> <src-dir> <src-dir> ..."
	exit 1;
fi

data=$1
shift
dest=$1
shift
first_src=$1

mkdir -p $dest/temp/log
cp $first_src/num_jobs $dest

for dir in $*; do
	src_id=$((src_id + 1))
	cur_num_jobs=$(cat $dest/num_jobs) || exit 1;
	lats=$(for n in $(seq $cur_num_jobs); do echo -n "$dir/lat.$n.gz "; done)
	$cmd $dir/log/copy_lattice.log \
	  lattice-copy "ark:gunzip -c $lats|" \
	  ark,scp:$dest/temp/lat.$src_id.ark,$dest/temp/lat.$src_id.scp || exit 1;
done

sort -m $dest/temp/lat.*.scp > $dest/temp/lat.scp

num_jobs=`cat $first_src/num_jobs`
echo "############## jobs = $num_jobs ################"

utils/split_data.sh $data $num_jobs

utils/filter_scps.pl JOB=1:$num_jobs \
  $data/split$num_jobs/JOB/utt2spk $dest/temp/lat.scp $dest/temp/lat.JOB.scp

for i in `seq 1 $num_jobs`; do
	sort -u $dest/temp/lat.${i}.scp > $dest/temp/lat.sort.${i}.scp
done

$cmd JOB=1:$num_jobs $dest/temp/log/combined_lattice.JOB.log \
	lattice-copy scp:$dest/temp/lat.sort.JOB.scp "ark:|gzip -c >$dest/lat.JOB.gz" || exit 1;

for f in phones.txt cmvn_opts final.alimdl final.mat final.mdl final.occs; do
	cp $first_src/$f $dest/$f
done

rm -r $dest/temp
