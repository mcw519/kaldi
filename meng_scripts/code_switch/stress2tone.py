import sys
import io
import re
import itertools as it

with io.open(sys.argv[1], 'r', encoding='utf-8') as f:
	file = [i.strip().split() for i in f.readlines()]


for i in range(0, len(file)):
	out = io.open(sys.argv[2], 'a+', encoding='utf-8')
	str = ' '.join(file[i][1:])
	digit_seq = re.findall(r"\d+", str)
	# check 1st stress at the end syllable mapping 1st stress to tone 4.
	if re.findall(r"\d+", str) != []:
		if digit_seq[len(digit_seq)-1] == '1':
			for index, phoneme in enumerate(file[i][1:]):
				if phoneme.find('1') != -1:
					last = index
			file[i][last+1] = file[i][last+1].replace('1', '4')
	else:
		pass

	# extend 
	dict = {}
	for index, phoneme in enumerate(file[i][1:]):
		if phoneme.find('0') != -1:
			base_phoneme = phoneme.replace('0', '')
			dict[index] = [base_phoneme+'0', base_phoneme+'3']
		elif phoneme.find('2') != -1:
			base_phoneme = phoneme.replace('2', '')
			dict[index] = [base_phoneme+'0', base_phoneme+'3']
		else:
			dict[index] = [phoneme]
	allNames = sorted(dict)
	combinations = it.product(*(dict[name] for name in dict))
	for prons in combinations:
		#print(prons)
		#print(dict)
		str = ' '.join(prons)
		out.write(file[i][0] + ' ' + str + '\n')
