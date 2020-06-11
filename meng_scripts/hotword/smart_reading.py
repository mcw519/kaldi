import sys
import io
import re


def check_md_char_illegal(x, words_table):
    '''
    read a 'list' and check each character.
    Returns:
        legal result with list format.
    '''
    flac = []
    for j in [a for a in ''.join(x)]:
        try:
            words_table[j]
            flac.append(True)
        except KeyError:
            flac.append(False)
    
    if False in flac:
        return False
    else:
        return True


def check_en_word_illegal(x, words_table):
    '''
    read 'list, dict' and check en word appear in words table.
    Returns:
        legal result with list format.
    '''
    try:
        words_table[x]
        return True
    except KeyError:
        return False


def smart_read(x, wd_table):
    '''
    Inout type: list, dict
    Format(x): <type id> <content>
        type 1:
            Chinese word, "每日一物", "吳孟哲"
        type 2:
            English word, "LEONA"
        type 3:
            Customize word pair, first column sould be your customize word and the others are how to spell.
            "TAYLOR-SWIFT TAYLOR SWIFT", "LMFAO L M F A O", "3D 3 D"
    Return:
        (int, string/list)
    '''

    _type = x[0].strip().split()[0]
    if _type == '1':
        _content = x[0].strip().split()[1]
        result = check_md_char_illegal(_content, wd_table)
        if result:
            return int(1), _content
        else:
            print("incorrect line", x)
            return int(1), False
    
    elif _type == '2':
        _content = x[0].strip().split()[1]
        result = check_en_word_illegal(_content, wd_table)

        if result:
            return int(2), _content
        else:
            print("incorrect line", x)
            return int(2), False
    
    elif _type == '3':
        _content = x[0].strip().split()[1:]
        result = []
        for i in _content[1:]:
            if re.search(r"[a-zA-Z]", i):
                result.append(check_en_word_illegal(i, wd_table))
            elif re.search(u'[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+', i):
                result.append(check_md_char_illegal(_content, wd_table))

        if False in result:
            print("incorrect line", x)
            return int(3), False
        else:
            return int(3), _content


def _test():
    words_table = {'TAYLOR': 1, 'SWIFT': 2, '每': 3, '日': 4, '一': 5, '物': 6, 'LEONA': 7, 'D': 8}
    hwd1 = ['1 每日一物']
    hwd2 = ['2 LEONA']
    hwd3 = ['3 TAYLOR-SWIFT TAYLOR SWIFT']
    hwd4 = ['3 LEONA-王 LEONA 王']
    
    for i in hwd1, hwd2, hwd3, hwd4:
        print(smart_read(i, words_table))


if __name__ == '__main__':
    _test()