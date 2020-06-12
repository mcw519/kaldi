# Copyright 2020  Meng Wu

import sys
import io

def s2h(seconds):
    m, s = divmod(float(seconds), 60)
    h, m = divmod(m, 60)
    s = round(s, 4)
    m = int(m)
    h = int(h)
    if s == 0.0:
        s = '00'
    elif s < 10:
        s = '0' + str(s)
    
    if m == 0.0:
        m = '00'
    elif m < 10:
        m = '0' + str(m)
    
    if h == 0.0:
        h = '00'
    elif h < 10:
        h = '0' + str(h)

    return('{h}:{m}:{s}'.format(h=h,m=m,s=s))

def ctm2str(x):
    '''
    Input:
      list with format <utt-id> <channel> <start-time> <duration> <word>
    Return:
      str file
    '''
    count = 5 # one line most have n words
    j = 0
    sequence = []
    for i in range(len(x)):
        this_key = x[i][0]
        w = io.open(this_key+'.srt', 'a+', encoding='utf-8') # write by utterance name
        start_time = x[j][2]
        end_time = float(x[i][2]) + float(x[i][3])
        if i != len(x) - 1 and str(end_time) == x[i+1][2] and len(sequence) < count:
            # this end == next start && sequence length <= count && not the end line
            sequence.append(x[i][4])
        elif len(sequence) == count: # one line most have n words
            sequence.append(x[i][4])
            start_time = s2h(start_time)
            end_time = s2h(end_time)
            w.write(start_time + ' --> '+ end_time + '\n' + ' '.join(sequence) + '\n' + '\n')
            sequence = []
            j = i + 1
        elif i != len(x) - 1 and (float(x[i+1][2]) - float(end_time)) <= 0.15 and len(sequence) < count:
            sequence.append(x[i][4])
        else:
            start_time = s2h(start_time)
            end_time = s2h(end_time)
            sequence.append(x[i][4])
            w.write(start_time + ' --> '+ end_time + '\n' + ' '.join(sequence) + '\n' + '\n')
            sequence = []
            j = i + 1





def main(path):
    with io.open(path, 'r', encoding='utf-8') as f:
        A = [ i.strip().split() for i in f.readlines() ]
    ctm2str(A)

if __name__ == '__main__':
    main(sys.argv[1])