import io
import copy
import kaldiio
import numpy as np

## read pdf id file format phone state pdf_1 pdf_2.

with io.open('temp', 'r') as f:
	a = [i.strip().split() for i in f.readlines()]

matrixM = list()
# total length  pdf id + 1(bias)
for i in range(288):
	letter = [0 for _ in range(289)] # add one dim for bias
	letter[i] = 0.5
	matrixM.append(letter)

matrixE = copy.deepcopy(matrixM)

for i in a:
	phone_id = int(i[0])
	pdf_id_1 = int(i[2])
	pdf_id_2 = int(i[3])
	if 3 < phone_id < 88:
		#print('EN phoneme')
		matrixE[pdf_id_1][pdf_id_1] = float(1)
		matrixE[pdf_id_2][pdf_id_2] = float(1)
		matrixM[pdf_id_1][pdf_id_1] = float(0)
		matrixM[pdf_id_2][pdf_id_2] = float(0)
	elif phone_id > 87:
		#print('MD phoneme')
		matrixE[pdf_id_1][pdf_id_1] = float(0)
		matrixE[pdf_id_2][pdf_id_2] = float(0)
		matrixM[pdf_id_1][pdf_id_1] = float(1)
		matrixM[pdf_id_2][pdf_id_2] = float(1)
		
# fix bias term
#matrixM[288][288] = float(0)
#matrixE[288][288] = float(0)

arrayM = np.array(matrixM)
arrayE = np.array(matrixE)
kaldiio.save_mat('M.mat', arrayM)
kaldiio.save_mat('E.mat', arrayE)
