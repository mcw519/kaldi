import io
import os
import sys
import string
import re

SPECIAL_SYMBOLS='：！，、「」？~（）～。”“๑• ؎ ˙…♫･∀ÒㅅÓ୧☉□୨《》『』●′∀‵ノ♡༼☯﹏☯༽´ ▽  ﾉ￣▽￣ ╯°Д°╯ ┻━┻ﾟωﾟ┛дΣ lli дﾟ΄◞ิ౪◟ิ ̀ ω ́ ✧ﾟдﾟ≡ﾟдﾟ ͡ ͜ʖ ͡ ╰ ́◉◞౪◟◉〃〃ˊωˋ ̀ώ・ω・◕ܫ◕ヾ╹◡╹ԅ≖‿≖ԅ・ಠಠつ ◕◕ つд⊙⎝ ◕д ◕ ⎠Σﾟдﾟε٩ gt ₃  lt ۶здಠಠ︵ ┳┳ ̀ ω ́ )d ･∀･)b♪ᴗ｡∩╮╭【】©෴©ლ꒪ヮ꒪ლᶘ ᴥᶅ🐔¯☜ヮ☜'


def open_file(path):
    with io.open(path, 'r', encoding='utf8') as f:
        # skip empty line
        a = list(line for line in (l.strip() for l in f) if line)
    f.close()
    return a


def is_segment(line):
    '''
    to generate segments file from srt format
    '''
    pattern = ' --> '
    if pattern in line:
        start, _, end = line.strip().split()
        s_h, s_m, s_s = start.split(':')
        e_h, e_m, e_s = end.split(':')
        # convert to seconds
        start_time =  3600 * float(s_h) + 60 * float(s_m) + float(s_s)
        end_time =  3600 * float(e_h) + 60 * float(e_m) + float(e_s)

        return(start_time, end_time)


def get_codemix_boundary(line):
    seg = []
    sub_line = ''
    flag = 'start'
    endpoint = len(line)
    count = 1
    for i in line:
        if flag == 'start':
            # initial
            if re.search('[a-zA-Z]', i):
                flag = 'en'
                sub_line = i
                count +=1
            else:
                flag = 'zh'
                sub_line = i
                count +=1
        elif count == endpoint:
            sub_line = sub_line + i
            seg.append(sub_line)
        elif not re.search('[a-zA-Z]', i) and flag == 'zh':
            # not english just appending
            sub_line = sub_line + i
            flag = 'zh'
            count +=1
        elif not re.search('[a-zA-Z]', i) and flag == 'en':
            # means mandarin word begine or between english puncation
            if i in string.punctuation:
                sub_line = sub_line + i
                count += 1
            elif i == '’':
                # typo correction
                sub_line = sub_line + '\''
                count += 1
            else:
                seg.append(sub_line)
                sub_line = i
                flag = 'zh'
                count +=1
        elif re.search('[a-zA-Z]', i) and flag == 'zh':
            # means English word begine
            seg.append(sub_line)
            sub_line = i
            flag = 'en'
            count +=1
        elif re.search('[a-zA-Z]', i) and flag == 'en':
            # means concate English word in one sub segments
            sub_line = sub_line + i
            count +=1
        else:
            seg.append(sub_line)
            count +=1
    return ' '.join(seg)
        

def remove_punc(line):
    line = get_codemix_boundary(line)
    punctuation = string.punctuation + SPECIAL_SYMBOLS
    line = line.strip().split()
    no_punc = []
    for word in line:
        if re.search('[a-zA-Z]', word):
            # contain english word
            try:
                while word[-1] in punctuation:
                    word = word[:-1]
                no_punc.append(word)
            except IndexError:
                pass
        else:
            # check each char in lines
            inside_string = ''
            prev = ''
            for i in range(len(word)):
                if i == len(word) - 1:
                    next = ''
                else:
                    next = word[i+1]

                if word[i] not in punctuation:
                    # not punctuation just append char
                    inside_string = inside_string + word[i]
                elif word[i] == '.' and re.search('[0-9]', prev) and re.search('[0-9]', next):
                    # reserve case 3.5 / 5.1234 , avoid 123.
                    inside_string = inside_string + word[i]
                else:
                    # replace punctuation with space
                    inside_string = inside_string + ' '
                prev = word[i]
            #for char in word:
            #    if char not in punctuation:
            #        inside_string = inside_string + char
            #    elif char == '.' and re.search('[0-9]', prev):
            #        # reserve case 3.5 / 5.1234
            #        inside_string = inside_string + char
            #    prev = char
            no_punc.append(inside_string)

    return ' '.join(no_punc)


def is_trans(line):
    '''
    to generate text file from srt format
    '''
    if 'WEBVTT' in line:
        pass
    elif 'Kind: captions' in line:
        pass
    elif 'Language: zh-TW' in line:
        pass
    elif 'Language: en' in line:
        pass
    elif ' --> ' in line:
        pass
    else:
        return(remove_punc(line))


def uttid_check(x):
    new = {}
    for line in x:
        key = line.strip().split()[0]
        body = line.strip().split()[1:]
        try:
            new[key].append(' '.join(body))
        except KeyError:
            new.update({key: body})
    # sorting
    #new = [(k,new[k]) for k in sorted(new.keys())]
    return new



def test_remove_punc():
    line="๑• ؎ •๑牙齒好冰啊 ˙˙顏市北七…呃…原句為：真地棒♫我是鐵頭功♫能開瓶蓋･∀･不要害我吃螺絲 ÒㅅÓ一飛沖天୧☉□☉୨讓媒體同業《放言》決定在19號的記者會『成蟲』素還真來囉●′∀‵ノ♡"
    print(remove_punc(line))

def main(path):
    '''
    Input:
        a srt file
    Return:
        two file for kaldi used
            1. segments with format <new_id> <old_id> <start_time> <end_time>
            2. text
    '''
    utt_name = os.path.basename(path).split('.')[0]
    out_seg = io.open(path + '.segments', 'a+', encoding='utf-8')
    out_text = io.open(path + '.text', 'a+', encoding='utf-8')
    f = open_file(path)
    count = 0
    pre_utt_id = ''
    for i in f:
        if is_trans(i):
            out_text.write(utt_name + '_' + str(count - 1).zfill(3) + ' ' + is_trans(i) + '\n')

        if is_segment(i):
            start, end = is_segment(i)
            out_seg.write(utt_name + '_' + str(count).zfill(3) + ' ' + utt_name + ' ' + str(start).zfill(3) + ' ' + str(end).zfill(3) + '\n')
            count = count + 1
    
    out_seg.close()
    out_text.close()

    segments = open_file(path + '.segments')
    text = open_file(path + '.text')

    try:
        assert len(text) > len(segments)
    except AssertionError:
        print('number of lines different between segments and text file')
        new_text = uttid_check(text)
        out_text = io.open(path + '.text2', 'a+', encoding='utf-8')
        for key, value in new_text.items():
            out_text.write(key + ' ' + ' '.join(value) + '\n')
        os.replace(path + '.text2', path + '.text')



if __name__ == '__main__':
    #test_remove_punc()
    main(sys.argv[1])