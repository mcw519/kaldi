import sys
import io

Vowel=["AA", "AA0", "AA1", "AA2", "AE", "AE0", "AE1", "AE2", "AH", "AH0", "AH1", "AH2", \
 "AO", "AO0", "AO1", "AO2", "AW", "AW0", "AW1", "AW2", "AY", "AY0", "AY1", "AY2", \
 "EH", "EH0", "EH1", "EH2", "ER", "ER0", "ER1", "ER2", "EY", "EY0", "EY1", "EY2", \
 "IH", "IH0", "IH1", "IH2", "IY", "IY0", "IY1", "IY2", "OW", "OW0", "OW1", "OW2", \
 "OY", "OY0", "OY1", "OY2", "UH", "UH0", "UH1", "UH2", "UW", "UW0", "UW1", "UW2"]

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

	# rule 1, check "AH0 N" in the end or not.
	if (W_P[len(W_P)-1] == "N" and W_P[len(W_P)-2] == "AH0"):
		W_P[len(W_P)-1] = "ern"
		W_P[len(W_P)-2] = "NULL" # NULL is a removed symbol.

	# special rule UN.
	if "UN" in W_P[0] and W_P[1] == "AH0" and W_P[2] == "N":
		W_P[1] = "a ng"
		W_P[2] = "NULL"

	# rule 3, N not present in front of vowel.
	for j in range(1, len(W_P)-1): # check until last 2 phoneme.
		if W_P[j] == "N" and W_P[j+1] not in Vowel:
			if W_P[j-1] == "AH0":
				W_P[j-1] = "NULL" # AH0 N merge to ern.
				W_P[j] = "ern"
			else:
				W_P[j] = "ern" # Voiced Consonants.

	# rule 2, N in the end.
	if W_P[len(W_P)-1] == "N":
		W_P[len(W_P)-1] = "ern"
	# check with suffix or not.
	if flag != 1:
		W_P.extend(keep_seq)
	str = ' '.join(W_P)
	out.write(str + '\n')



