import io
import sys
import random

with io.open(sys.argv[1] + '/wav.scp', 'r', encoding='utf-8') as f:
	wav1 = [i.strip().split() for i in f.readlines()]
with io.open(sys.argv[2] + '/wav.scp', 'r', encoding='utf-8') as f:
        wav2 = [i.strip().split() for i in f.readlines()]
with io.open(sys.argv[1] + '/text_1', 'r', encoding='utf-8') as f:
	text1_1 = [i.strip().split() for i in f.readlines()]
with io.open(sys.argv[1] + '/text_2', 'r', encoding='utf-8') as f:
        text1_2 = [i.strip().split() for i in f.readlines()]
with io.open(sys.argv[2] + '/text', 'r', encoding='utf-8') as f:
        text2 = [i.strip().split() for i in f.readlines()]



for i in range(0, len(wav1)):
#for i in range(0, 1):
	wav_out = io.open('wav.scp', 'a+', encoding='utf-8')
	text_out_1 = io.open('text_1', 'a+', encoding='utf-8')
	text_out_2 = io.open('text_2', 'a+', encoding='utf-8' )
	
	utt_1_id = wav1[i][0]
	utt_1_path = wav1[i][1]
	rand = random.randint(0, len(wav2)-1)
	utt_2_id = wav2[rand][0]
	utt_2_path = wav2[rand][1]
	
	wav_out.write(utt_2_id + '__' + utt_1_id + ' sox -M ' + utt_1_path + ' ' + utt_2_path + ' -r 16000 -t wav - |' + '\n')
	text_out_1.write(utt_2_id + '__' + utt_1_id + ' ' + ' '.join(text2[rand][1:]) + ' ' + ' '.join(text1_1[i][1:]) + '\n')
	text_out_2.write(utt_2_id + '__' + utt_1_id + ' ' + ' '.join(text2[rand][1:]) + ' ' + ' '.join(text1_2[i][1:]) + '\n')
	wav_out.write(utt_1_id + '__' + utt_2_id + ' sox -M ' + utt_2_path + ' ' + utt_1_path + ' -r 16000 -t wav - |' + '\n')
	text_out_1.write(utt_1_id + '__' + utt_2_id + ' ' + ' '.join(text1_1[i][1:]) + ' ' + ' '.join(text2[rand][1:]) + '\n')
        text_out_2.write(utt_1_id + '__' + utt_2_id + ' ' + ' '.join(text1_2[i][1:]) + ' ' + ' '.join(text2[rand][1:]) + '\n')
