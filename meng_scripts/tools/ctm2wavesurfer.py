import io
import sys
with io.open(sys.argv[1], 'r' , encoding = 'utf-8') as f:
    words = [i.strip().split() for i in f.readlines()]

for i in range(0,len(words)):
        out = io.open(words[i][0],'a+',encoding = 'utf-8')
        out.write(words[i][2]+" "+unicode(float(words[i][3])+float(words[i][2]))+" "+words[i][4]+'\n')

