import sys
import io
import re

Vowel=["AA", "AA0", "AA1", "AA2", "AA3", "AA4", "AE", "AE0", "AE1", "AE2", "AE3", "AE4", "AH", "AH0", "AH1", "AH2","AH3", "AH4" \
 "AO", "AO0", "AO1", "AO2", "AO3", "AO4", "AW", "AW0", "AW1", "AW2", "AW3", "AW4", "AY", "AY0", "AY1", "AY2", "AY3", "AY4" \
 "EH", "EH0", "EH1", "EH2", "EH3", "EH4", "ER", "ER0", "ER1", "ER2", "ER3", "ER4", "EY", "EY0", "EY1", "EY2", "EY3", "EY4" \
 "IH", "IH0", "IH1", "IH2", "IH3", "IH4", "IY", "IY0", "IY1", "IY2", "IY3", "IY4", "OW", "OW0", "OW1", "OW2", "OW3", "OW4" \
 "OY", "OY0", "OY1", "OY2", "OY3", "OY4", "UH", "UH0", "UH1", "UH2", "UH3", "UH4", "UW", "UW0", "UW1", "UW2", "UW3", "UW4", "NULL"]

AO=["AO", "AO0", "AO1", "AO2", "AO3", "AO4"]

with io.open(sys.argv[1], 'r', encoding='utf-8') as f:
	line = [i.strip().split() for i in f.readlines()]

for i in range(0, len(line)):
	flag=1
	out = io.open(sys.argv[2], 'a+', encoding='utf-8')
	W_P = line[i]
	# suffix check, ex:'s ....
	if "'S" in W_P[0] and W_P[len(W_P)-1] == "Z":
		W_P_replace=W_P[:len(W_P)-1]
		keep_seq=W_P[len(W_P)-1]
		W_P=W_P_replace
		flag=2

	# rule 1, check "AH0 L" in the end or not.
	if (W_P[len(W_P)-1] == "L" and W_P[len(W_P)-2] == "AH0"):
		W_P[len(W_P)-1] = "o0 u0"
		W_P[len(W_P)-2] = "NULL" # NULL is a removed symbol.
	elif (W_P[len(W_P)-1] == "L" and W_P[len(W_P)-2] == "AH3"):
		W_P[len(W_P)-1] = "o3 u3"
		W_P[len(W_P)-2] = "NULL"
		
	# rule 2, L in the end.
	#if W_P[len(W_P)-1] == "L":
	#	W_P[len(W_P)-1] = "o u"

	# rule 3, L not present in front of vowel.
	for j in range(1, len(W_P)-1): # check until last 2 phoneme.
		if W_P[j] == "L" and W_P[j+1] not in Vowel:
			if W_P[j-1] == "AH0":
				W_P[j-1] = "NULL" # AH0 L merge to "o u".
				W_P[j] = "o0 u0"
			elif W_P[j-1] == "AH3":
                                W_P[j-1] = "NULL" # AH0 L merge to "o u".
                                W_P[j] = "o3 u3"
			elif W_P[j-1] == "AO0": # special rule.
				W_P[j-1] = "NULL"
				W_P[j] = "o0 u0"
			elif W_P[j-1] == "AO3": # special rule.
                                W_P[j-1] = "NULL"
                                W_P[j] = "o3 u3"
			else:
				W_P[j] = "o3 u3" # Voiced Consonants.
	
	# rule 2, L in the end.
	if W_P[len(W_P)-1] == "L" and W_P[len(W_P) -2] not in AO:
		pre_tone = ' '.join(re.findall(r"\d+", W_P[len(W_P)-2]))
		if pre_tone == "0":
                        W_P[len(W_P)-1] = "o0 u0"
                elif pre_tone == "1":
                        W_P[len(W_P)-1] = "o1 u1"
                elif pre_tone == "3":
                        W_P[len(W_P)-1] = "o3 u3"
	elif W_P[len(W_P)-1] == "L" and W_P[len(W_P) -2] == "AO4": # avoid "AO L" case.
		W_P[len(W_P)-2] = "NULL"
		W_P[len(W_P)-1] = "o4 u4"
	elif W_P[len(W_P)-1] == "L" and W_P[len(W_P) -2] == "AO3": # avoid "AO L" case.
                W_P[len(W_P)-2] = "NULL"
                W_P[len(W_P)-1] = "o3 u3"
	elif W_P[len(W_P)-1] == "L" and W_P[len(W_P) -2] == "AO1": # avoid "AO L" case.
                W_P[len(W_P)-2] = "NULL"
                W_P[len(W_P)-1] = "o1 u1"
	elif W_P[len(W_P)-1] == "L" and W_P[len(W_P) -2] == "AO0": # avoid "AO L" case.
                W_P[len(W_P)-2] = "NULL"
                W_P[len(W_P)-1] = "o0 u0"

	# check with suffix or not.
	if flag != 1:
		keep_seq 
		W_P.extend(keep_seq)
	str = ' '.join(W_P)
	out.write(str + '\n')



