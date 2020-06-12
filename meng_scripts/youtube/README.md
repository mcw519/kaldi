# YouTube data collect

## YouTube data download by youtube-dl tool

Install youtube-dl and ffmpeg
```
sudo pip install --upgrade youtube_dl
sudo apt-get install ffmpeg
```

Download whole youtube channel audios
```
youtube-dl -f 140 -i --id --all-subs <youtube playlist url>
```

Modify download dir to Kaldi format and prepare folder for cleanup used
```
run_prepare_download_dir.sh <download_dir>
run_prepare_dir_to_run.sh <download_dir>
```

Run Cleanup by confidence island decoding
```
run_make_utt_linear_fsts.sh <download_dir> <AM_model>
decode_confidence_island.sh <download_dir>/graph <download_dir> <decode_dir>
run_prepare_cleanup_dir.sh <download_dir> <confidence_island_dir>
```

## reference
[LARGE SCALE DEEP NEURAL NETWORK ACOUSTIC MODELING WITH SEMI-SUPERVISED TRAINING DATA FOR YOUTUBE VIDEO TRANSCRIPTION](https://static.googleusercontent.com/media/research.google.com/zh-TW//pubs/archive/41403.pdf)
