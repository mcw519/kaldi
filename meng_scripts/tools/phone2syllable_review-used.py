import io
import re
import sys

def get_boundary(phone_seq):
	match_list = ['a0', 'a1', 'a2', 'a3', 'a4', 'e0', 'e1', 'e2', 'e3', 'e4', 'er0', 'er1', 'er2', 'er3', 'er4', 'ern0', 'ern1', 'ern2', 'ern3', 'ern4', 'err0', 'err1', 'err2', 'err3', 'err4', 'i0', 'i1', 'i2', 'i3', 'i4', 'ii0', 'ii1', 'ii2', 'ii3', 'ii4', 'ng0', 'ng1', 'ng2', 'ng3', 'ng4', 'nn0', 'nn1', 'nn2', 'nn3', 'nn4', 'u0', 'u1', 'u2', 'u3', 'u4', 'yu0', 'yu1', 'yu2', 'yu3', 'yu4', 'o1', 'o2', 'o3', 'o4']
	phone_list = phone_seq.strip().split()
	first_boundary = []
	sub_segments = []
	for key, phoneme in enumerate(phone_list):
		# 1. check consonan place.
		if phoneme not in match_list:
			first_boundary.append(key)
	# 2. get first time segments.
	if first_boundary != [] and first_boundary[0] != 0: # one syllable word would crash.
		num_seg = len(first_boundary) + 1
		for i in range(0, num_seg):
			if i == 0:
				key = first_boundary[0]
				sub_segments.append(phone_list[0:key])
			elif i == len(first_boundary):
				key = first_boundary[i-1]
				sub_segments.append(phone_list[key:])
			else:
				key_1 = first_boundary[i-1]
				key_2 = first_boundary[i]
				sub_segments.append(phone_list[key_1:key_2])
	else:
		num_seg = len(first_boundary)
		for i in range(0, num_seg):
			if i == 0 and num_seg != 1:
				key = first_boundary[i+1]
				sub_segments.append(phone_list[0:key])
			elif i == 0 and num_seg == 1:
				sub_segments.append(phone_list[0:])
			elif i == len(first_boundary) - 1:
				key = first_boundary[i]
				sub_segments.append(phone_list[key:])
			else:
				key_1 = first_boundary[i]
				key_2 = first_boundary[i+1]
				sub_segments.append(phone_list[key_1:key_2])
	# 3. get per segments check.
	#     a. check same tone and same phoneme.
	#     b. check different tone.
	#     c. check only one final in one segments.
	new_sub_segments_a = []
	for i in sub_segments:
		key = 0
		tag = '0'
		for j in range(0, len(i)):
			if j+1 != len(i) and i[j] == i[j+1]:
				new_sub_segments_a.append(i[key:j+1])
				key = j + 1
				tag = '1'
			elif j+1 == len(i) and tag != '0':
				new_sub_segments_a.append(i[key:])
		if tag == '0':
			new_sub_segments_a.append(i)
	new_sub_segments_b = []
	for segment in new_sub_segments_a:
		str = []
		base_tone = []
		for i in segment:
			this_tone = re.findall(r'\d', i)
			if this_tone == []:
				str.append(i)
			elif base_tone == [] and this_tone != []: # c a1
				base_tone = this_tone
				str.append(i)
			elif base_tone != [] and this_tone != base_tone: # tone change a1 a2/ a1 c.
				base_tone = this_tone
				new_sub_segments_b.append(str)
				str = [] # create new str
				str.append(i)
			else:
				str.append(i)
		new_sub_segments_b.append(str)
	#print(new_sub_segments_b)
	# 4. some phoneme must be in the syllable end.
	end_point = ['nn1', 'nn2', 'nn3', 'nn4', 'ng0','ng1', 'ng2', 'ng3', 'ng4','ern0', 'ern1', 'ern2', 'ern3', 'ern4', 'err1', 'err2', 'err3', 'err4']
	new_sub_segments_c = []
	for segment in new_sub_segments_b:
		str = []
		for i in segment:
			if i not in end_point:
				str.append(i)
			else:
				str.append(i)
				new_sub_segments_c.append(str)
				str = []
		new_sub_segments_c.append(str)
	new_sub_segments_c[:] = (value for value in new_sub_segments_c if value != [])
	length = len(new_sub_segments_c)
	return new_sub_segments_c, length


# main
# create two file, one is already word2syllable, the other is partial syllable which need to be check.
with io.open(sys.argv[1], 'r', encoding='utf-8') as f:
	lex = [i.strip().split() for i in f.readlines()]
with io.open(sys.argv[2], 'r', encoding='utf-8') as f: # load syllable table store in dict.
	aa = [i.strip().split() for i in f.readlines()]

table = {}
for i in range(0, len(aa)):
	table[' '.join(aa[i][1:])] = aa[i][0]

for i in range(0, len(lex)):
	out = io.open(sys.argv[3], 'a+', encoding='utf-8')
	no_work = io.open(sys.argv[4], 'a+', encoding='utf-8')
	phone_seq = ' '.join(lex[i][1:])
	print('line: ' + str(i))
	if len(lex[i][1:]) == len(lex[i][0]): # phoneme sequence == word sequence. one phoneme = one syllable
		print('skip line' + ' ' + str(i))
		syl_str = ''
		for j in range(0, len(lex[i][1:])):
			aa = table[lex[i][j+1]]
			syl_str = syl_str + ' ' + aa
		out.write(lex[i][0] + ' ' + syl_str + '\n')
	elif len(lex[i][0]) == 1: # only one syllable word don't need to run the program.
		phone_seq = ' '.join(lex[i][1:])
		syl_str = ''
		syl_str = table[phone_seq]
		if phone_seq in table:
			out.write(lex[i][0] + ' ' + syl_str + '\n')
		else:
			no_work.write(lex[i][0] + ' ' + ' '.join(lex[i][1:]) + '\n')
	else:
		seq, length = get_boundary(phone_seq)
		if length == 0:
			no_work.write(lex[i][0] + ' ' + '{'+ ' '.join(lex[i][1:]) + '}\n')
		elif length != len(lex[i][0]) and length != 0:
			syl_str = ''
			for j in range(0, length):
				phone_seq = ' '.join(seq[j][:])
				if phone_seq in table:
					aa = table[phone_seq]
					syl_str = syl_str + ' ' + aa
				else:
					syl_str = syl_str + ' ' + '{' + phone_seq + '}'
			no_work.write(lex[i][0] + ' ' + syl_str + '\n')
		else:
			syl_str = ''
			for j in range(0, length):
				phone_seq = ' '.join(seq[j][:])
				aa = table[phone_seq]
				syl_str = syl_str + ' ' + aa
			out.write(lex[i][0] + ' ' + syl_str + '\n')
