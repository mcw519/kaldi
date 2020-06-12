# Copyright 2020  Meng Wu

import sys
import io
import os
from edit_distance import edit_distance

def ctm2dict(ctm):
    '''
    Input:
        ctm file
    Return:
        two dict (start dict, dur dict)
    '''
    start_dict = {}
    dur_dict = {}
    with io.open(ctm, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            key = line.strip().split()[0]
            start_value = [line.strip().split()[2]]
            dur_value = [line.strip().split()[3]]
            
            try:
                start_dict[key].append(start_value)
                dur_dict[key].append(dur_value)
            except KeyError:
                start_dict.update({key: [start_value]})
                dur_dict.update({key: [dur_value]})
    
    return start_dict, dur_dict

def confidence_island(hyp, ref, island=5, position=True):
    '''
    Input:
        hyp, ref, island count, position [list, list, int, bool]
    Return:
        list
    '''
    _, result, trans, _ = edit_distance(ref, hyp)
    result = result.strip().split()[1:]
    trans = trans.strip().split()[1:]
    assert len(result) == len(trans)
    shift = 0
    bias = 0
    seg = []
    if position:
        flag = 0
        pos = []
    
    temp_wd = []
    for i in range(len(result)):
        if result[i] != 'C':
            seg.append(temp_wd + [len(temp_wd)] )
            temp_wd = [] # new segment list
            if position:
                pos.append([flag , i - bias])
                flag = i - bias
                if trans[i] == '&&':
                    bias += 1
                if trans[i] == '$$':
                    flag += 1
                if trans[i] == '**':
                    flag += 1
        elif i == len(result) - 1 :
            temp_wd.append(trans[i])
            seg.append(temp_wd + [len(temp_wd)])
            if position:
                pos.append([flag , i - bias + 1])
        else:
            temp_wd.append(trans[i])
    final = []

    if position:
        final_pos = []
    for i in range(len(seg)):
        if int(seg[i][-1]) >= int(island):
            final.append(seg[i])
            if position:
                final_pos.append([pos[i]])
    if position:
        return final, final_pos
    else:
        return final
        

def generate_new_segment(s_dict, d_dict, uttid, boundary):
    '''
    Input:
        start dict, duration_dict, uttid, boundary [dict, dict, str, list]
    Return:
        Kaldi segments file format, <seg-uttid> <ori-uttid> <satrt-time> <end-time> [list]
    '''
    start_dict = s_dict
    dur_dict = d_dict
    if len(boundary) > 1:
        # one line include more than 1 segment: uttid_segxxx
        j = 0
        for i in boundary:
            ss, ee = i
            new_uttid = uttid + str('_seg') + str(j).zfill(3)
            start = start_dict[uttid][ss:ee][0][0]
            end = float(start_dict[uttid][ss:ee][-1][0]) + float(dur_dict[uttid][ss:ee][-1][0])
            j += 1
            return new_uttid, uttid, str(start), str(end)
    else:
        ss, ee = boundary[0]
        start = start_dict[uttid][ss:ee][0][0]
        end = float(start_dict[uttid][ss:ee][-1][0]) + float(dur_dict[uttid][ss:ee][-1][0])
        return uttid, uttid, str(start), str(end)




def _test_edit():
    hyp = ['幫我', '設定', '早上', '的', '鬧鐘', '吧']
    ref = ['幫我', '設定', '明天', '早上', '的', '鬧鐘']
    distance, result, trans, opt_wd = edit_distance(hyp, ref)
    print(distance)
    print(result)
    print(trans)
    print(opt_wd)


def _test_island():
    #hyp = ['幫我', '設定', '早上', '的', '鬧鐘', '吧']
    #ref = ['請', '幫我', '設定', '明天', '早上', '的', '鬧鐘']
    #ref = [ '我', '想聽', '八三夭', '的', '拍手', '歌']
    #hyp = [ '我', '想聽', '八三夭', '的', '拍手' ]
    #hyp = ['請問', '現在', '大三']
    #ref = ['請問', '現在', '大三']
    #ref = [ '幫我', '設定', '六點半' ]
    #hyp = [ '幫我', '設定', '早上', '六點半' ]
    hyp = ['幫我', '設定', '早', '的', '鬧鐘', '吧']
    ref = [ 'A', 'B', 'C', '幫我', '設定', '後天', 'A', 'B', 'C', "A's", '早上', '的', '鬧鐘' ]
    #hyp = ['寫', '放入', '混合疫苗', '有機的', '地', '在', '底片', '上', '放入', '檢測', '平台', '十二分鐘', '就能', '驗出', '是否', '感染', '中共', '肺炎', '台灣', '新創', '公司', '研發', '的', '產品', '已經', '在', '醫院', '執行', '初步', '臨床', '驗證', '靈敏度', '高達', '九成', 'THEY', 'GAVE', 'US', 'FIFTEEN', 'PATIENCE', 'SAMPLES', 'TEN', 'OF', 'THEM', 'WERE', 'HOSPITALIZED', 'PCR', 'CONFIRM', 'PATIENTS', 'ARE', 'FIVE', 'OF', 'THEM', 'WERE', 'CONTROL', 'SO', 'NEGATIVE', 'SAMPLE', 'AND', 'THEY', 'DID', 'NOT', 'TELL', 'US', 'WHICH', 'ONE', 'WAS', 'WHICH', 'SO', 'WE', 'MEASURED', 'THEM', 'WITH', 'THE', 'BLUE', 'BOXING', 'AND', 'WE', 'COULD', 'FIND', 'THEIR', 'NINE', 'OUT', 'OF', 'TEN', 'OF', 'THE', 'POSITIVE', 'AND', 'ALL', 'THE', 'FIVE', 'NEGATIVE', 'STIRRED', 'NEGATIVE', 'DANISH', 'ATTESTS', 'SEEMS', 'VERY', 'PROMISING', 'ENORMOUSLY', '回流', '到', '技術', '再加上', '吃', '豬肝', 'GUN', '磁場', '的', '反應', '所以', '可以', '把', '這個', '所謂', '的', '反應', '時間', '整個', '家', '就是', '撿到', '三十分鐘', '以內', '這回', '血清', '檢測', '能夠', '同時', '檢測', '跟', '台', 'GG', '能夠', '檢測', 'EASY', '無症狀', '患者', '童子軍', '病患', '是否', '已經', '沒有', '傳染力', '符合', '出言', '標準', '這種', '一般', '的', '核酸', '檢測', '需要', '耗費', '四', '個', '小時', '就會', '檢測', '工具', '只需', '十二分鐘', '攜帶', '方便', '適合', '用在', '機場', '港口', '等', '第一線', '防疫', '場所', '有機會', '六月', '在', '台灣', '上市', '預計', '下週', '會', '帶來', "LUIGI'S", '開始', '更大', '的', '一個', '臨床', '驗證', '如果', '順利', '同步', '在', '歐盟', '認證', '如果', '順利', '的話', '預計', '在', '五月', '多', '左右', '應該', '就可以', '讓', '產品', '上市', '公司', '擁有', '國際', '團隊', '研發', '基地', '座落', '在', '丹麥', '中心', '等', '設備', '生產', '在', '台灣', '透過', '跨國', '技術合作', '期盼', '能', '協助', '檢測', '量', '能不', '足', '的', '國家', '地區', '對抗', '疫情']
    #ref = ['採', '一', '滴血', '放入', '混合液', '搖', '一', '搖', '接著', '滴', '在', '碟片', '上', '放入', '醫療', '檢測', '平台', '十二分鐘', '就', '能', '驗出', '是否', '感染', '中共', '肺炎', '台灣', '新創', '公司', '研發', '的', '這', '款', '檢測', '產品', '已經', '在', '丹麥', '第二', '大', '醫院', '執行', '初步', '臨床', '驗證', '靈敏度', '高達', '九成', 'THEY', 'GAVE', 'US', 'FIFTEEN', 'PATIENT', 'SAMPLES', 'TEN', 'OF', 'THEM', 'WERE', 'HOSPITAL', 'ANALYZED', 'P.C.R.', 'CONFIRMED', 'PATIENT', 'FIVE', 'OF', 'THEM', 'WERE', 'CONTROLLED', 'NEGATIVE', 'SAMPLE', 'AND', 'THEY', 'DID', 'NO', 'TELL', 'US', 'WHICH', 'ONE', 'WAS', 'WHICH', 'SO', 'WE', 'MEASURE', 'THEM', 'WITH', 'THE', 'BULL', 'BOX', 'AND', 'WE', 'COULD', 'FIND', 'THE', 'NINE', 'OUT', 'OF', 'TEN', 'OF', 'POSITIVE', 'AND', 'ALL', 'THE', 'FIVE', 'NEGATIVES', 'ARE', 'NEGATIVE', 'THE', 'INITIAL', 'TESTING', 'VERY', 'PROMISING', '因為', '我們', '是', '用', '微', '流', '道', '的', '技術', '再', '加上', '磁珠', '跟', '磁場', '的', '反應', '所以', '可以', '把', '這個', '所謂', '的', '反應', '時間', '整個', '加', '就是', '減', '到', '剩', '十分鐘', '以內', '這', '款', '血清', '檢測', '能夠', '同時', '檢測', 'I.G.M.', '跟', 'I.G.G.', '能', '個', '檢測', '疑似', '無症狀', '患者', '同時', '確認', '病患', '是否', '已經', '沒有', '傳染力', '符合', '出院', '標準', '相', '較', '一般', '的', '核酸', '檢測', '需要', '耗費', '四', '個', '小時', '這', '款', '檢測', '工具', '只', '需', '十二分鐘', '攜帶', '操作', '方便', '適合', '用', '在', '機場', '港口', '等', '第一線', '防疫', '場所', '有', '機會', '六月', '在', '台灣', '上市', '預計', '是', '在下', '週', '就會', '在', '義大利', '舉', '呃', '那個', '開始', '更', '大', '的', '一個', '臨床', '驗證', '那', '如果', '順利', '的話', '因為', '我們', '同步', '在', '準備', '那個', 'C.E.', 'MARK', '就是', '歐盟', '的', '認證', '如果', '順利', '的話', '預計', '是', '在', '五月', '多', '左右', '應該', '就', '可以', '讓', '產品', '上市', '台灣', '新創', '公司', '擁有', '國際', '團隊', '生化', '研發', '基地', '坐落', '在', '丹麥', '軟硬體', '研發中心', '跟', '設備', '生產', '則', '位在', '台灣', '透過', '跨國', '技術合作', '期盼', '能', '協助', '檢測', '量', '能', '不足', '的', '國家', '地區', '對抗', '疫情']
    #result, position = confidence_island(hyp, ref, island=2, position=True)
    island, pos = confidence_island(hyp, ref, island=1, position=True)
    #print(hyp)
    print(island, pos)
    idx = 0
    for i in pos:
        #print(i)
        ss, ee = i[0]
        assert hyp[ss:ee] == island[idx][0:-1]
        idx += 1
        print(hyp[ss:ee])
    #print(result, position)


def main(ref, hyp, ctm):
    '''
    Input data format: <utt-id> <trans> ....
    '''
    count=3
    folder_path = os.path.abspath('/'.join(hyp.split('/')[:-1]))
    with io.open(ref, 'r', encoding="utf-8") as f:
        ref = [ i.strip().split() for i in f.readlines() ]
    with io.open(hyp, 'r', encoding="utf-8") as f:
        hyp = [ i.strip().split() for i in f.readlines() ]
    
    os.mkdir(folder_path + '/confidence_island')
    island = io.open(folder_path + '/confidence_island/island_text', 'a+', encoding='utf-8')

    if ctm != 'None':
        s_dict, d_dict = ctm2dict(ctm)
        segments = io.open(folder_path + '/confidence_island/segments', 'a+', encoding='utf-8')

    assert len(ref) == len(hyp)
    for i in range(len(ref)):
        assert ref[i][0] == hyp[i][0]
        if len(hyp[i]) == 1:
            hyp_trans = ['']
        else:
            hyp_trans = hyp[i][1:]

        if len(ref[i]) == 1:
            ref_trans = ['']
        else:
            ref_trans = ref[i][1:]

        if ctm != 'None':
            result, boundary = confidence_island(hyp_trans, ref_trans, island=count, position=True)
        else:
            result = confidence_island(hyp_trans, ref_trans, island=count, position=False)

        if ctm != 'None':
            if len(result) < 1:
                pass
            elif len(result) >= 2:
                c = 0
                for seg in result:
                    new_id = ref[i][0] + '_seg' + str(c).zfill(3)
                    _, temp2, temp3, temp4 = generate_new_segment(s_dict, d_dict, ref[i][0], boundary[c])
                    island.write(new_id + ' ' + ' '.join(seg[:-1]) + '\n')
                    segments.write(new_id + ' ' + str(temp2) + ' ' +  str(temp3) + ' ' + str(temp4) + '\n')
                    c += 1
            else:
                _, temp2, temp3, temp4 = generate_new_segment(s_dict, d_dict, ref[i][0], boundary[0])
                island.write(ref[i][0] + ' ' + ' '.join(result[0][:-1]) + '\n')
                segments.write(ref[i][0] + ' ' + str(temp2) + ' ' +  str(temp3) + ' ' + str(temp4) + '\n')


        else:
            if len(result) < 1:
                pass
            elif len(result) >= 2:
                c = 0
                for seg in result:
                    new_id = ref[i][0] + '_seg' + str(c).zfill(3)
                    island.write(new_id + ' ' + ' '.join(seg[:-1]) + '\n')
                    c += 1
            else:
                island.write(ref[i][0] + ' ' + ' '.join(result[0][:-1]) + '\n')


if __name__ == '__main__':
    #_test_edit()
    #_test_island()
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], ctm=sys.argv[3])
    else:
        main(sys.argv[1], sys.argv[2], ctm='None')
