import sys
import io
import re

## input is opt

with io.open(sys.argv[1], 'r', encoding = 'utf-8') as f:
	A = [i.strip().split() for i in f.readlines()]

wd_count=0
d_count=0
s_count=0
i_count=0
c_count=0
for i in range(0, len(A)):
	#print(A[i])
	if re.search(r"[A-Z]", A[i][1]):
		if re.search(r"^d", A[i][0]):
			d_count += int(A[i][3])
		elif re.search(r"^s", A[i][0]):
                        s_count += int(A[i][3])
		elif re.search(r"^c", A[i][0]):
                        c_count += int(A[i][3])
	if re.search(r"[A-Z]", A[i][2]):
		if re.search(r"^i", A[i][0]):
                        i_count += int(A[i][3])

wrong=d_count+s_count+i_count
total_wd=d_count+s_count+c_count

print("EN-WER: " + str(float(wrong)/float(total_wd)))
print("deletion: " + str(d_count))
print("subtitution: " + str(s_count))
print("inseartion: " + str(i_count))
print("correct: " + str(c_count))
