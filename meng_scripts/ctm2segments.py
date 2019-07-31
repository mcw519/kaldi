import sys
import io

with io.open(sys.argv[1], 'r', encoding='utf-8') as f:
	line = [i.strip().split() for i in f.readlines()]

pre_id = ''
suffix = 0
for i in range(0, len(line)):
	if line[i][4] != '<SPOKEN_NOISE>':
		utt_id = line[i][0]
		start = line[i][2]
		end = float(line[i][2]) + float(line[i][3])
		word = line[i][4]
		if utt_id == pre_id:
			suffix = suffix + 1
			uid = str(utt_id) + '_' + str(suffix)
			string = uid + ' ' + utt_id + ' ' + start + ' ' + str(end)
#			print(string)
			print(string+' '+word)
		else:
			suffix = 0
			pre_id = utt_id
			uid = str(utt_id) + '_' + str(suffix)
                        string = uid + ' ' + utt_id + ' ' + start + ' ' + str(end)
			#print(string)
                        print(string+' '+word)


