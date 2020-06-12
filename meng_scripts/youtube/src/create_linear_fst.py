# Copyright 2020  Meng Wu

import sys
import io
from hotword_context import ChineseSegmenter
from hotword_context import read_file_to_dict

LEX_PATH = '/var/ctc5nlp/meng/AICS_ASR/feedback_loop/confidence_island/src/data/lang_20200102/words.txt'

'''
Usage: 
    multi line with file: 
        python create_linear_fst.py text [vocab.prob] | fstcompile > G.fst
    single line with stdin:
        cat text | python create_linear_fst.py | fstcompile > G.fst
'''

if len(sys.argv) > 1:
    with io.open(sys.argv[1], 'r', encoding='utf-8') as f:
        trans = [ i.strip().split()[1:] for i in f.readlines() ]
    lex = read_file_to_dict(LEX_PATH)
else:
    # single utterance
    trans = [[ i for i in sys.stdin.readline().strip().split()[1:] ]]
    lex = read_file_to_dict(LEX_PATH)

print('0')
num = 0
for i in trans:
    if i == '':
        # pass empty line
        pass
    else:
        if len(sys.argv) >= 2:
            text, _ = ChineseSegmenter(''.join(i), sys.argv[2], character_base=False)
        else:
            text = i
        for j in range(len(text)):
            try:
                key = text[j]
                _ = lex[key]
            except KeyError:
                text[j] = '<SPOKEN_NOISE>' # map oov as <SPOKEN_NOISE>
            num = num + 1
            if j == 0:
                print('0\t{next}\t{wd}\t{wd}'.format(next=num, wd=text[j]))
            else:
                print('{start}\t{next}\t{wd}\t{wd}'.format(start=num-1, next=num, wd=text[j]))
            print('0\t{next}\t<eps>\t<eps>'.format(next=num))
            print(num)
