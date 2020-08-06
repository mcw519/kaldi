#! /usr/bin/enve/ python3

# Copyright 2020 (author: Meng Wu)

import os
import io
import math
import re


def read_file_to_list(x):
  with io.open(x, 'r', encoding='utf-8') as f:
    a = [ i.strip().split() for i in f.readlines() ]

  return a


def read_file_to_dict(x):
  dict_final = {}
  with io.open(x, 'r', encoding='utf-8') as f:
    for i in f.readlines():
      key = i.strip().split()[0]
      value = i.strip().split()[1]
      dic = {key: value}
      dict_final.update(dic)
  
  return dict_final


def check_oov_and_merge(word, words_table):
  '''
    check oov and merge oov into words.txt.
    Returns: dict
  '''

  count = len(words_table) - 1
  try:
    # if word in word_table don't need to add new word id.
    _ = words_table[word]
  except KeyError:
    count = count + 1
    dic = {word: str(count)}
    words_table.update(dic)
  
  return words_table


def ChineseSegmenter(x, words_prob, character_base='false'):
  '''
    This is a word segment function.
    Input:
        x : words string.
        words_prob : words table with each word frequence.
        character_base=True/False : Return character based segment result.
    Returns:
        list.
  '''
  wd_prob = read_file_to_dict(words_prob)
  default_weight = float(100)
  assert len(x) > 0, 'be sure input string not empty'
  text_len = len(x)
  best_prob = [float(0)] # inital 
  prev = []
  current_weight = float(0)
  
  for i in range(text_len):
    for j in range(i+1, text_len+1):
      # exceeds the max length
      if j - i <= text_len:
        # matching an entry in the vocabulary
        current_wd = x[i:j]
        try:
          wd_prob[current_wd]
        except KeyError:
          continue

      # get the previous found path, if not exists, use the default value,
      # which means we may take the previous token as the path.
      pre_weight = float(0)
      if i > 0:
        try:
          best_prob[i] # start from i=1
          pre_weight = best_prob[i]
        except IndexError:
          pre_weight = default_weight

      # calculate weight for curent path
      w = float(wd_prob[current_wd])
      current_weight = pre_weight + float(w)

      # update path
      try:
        prev[j-1] # !exists($prev[$j])
        if best_prob[j] > current_weight:
          prev[j] = i
          best_prob[j] = current_weight
      except IndexError:
        prev.append(i)
        best_prob.append(current_weight)
      j = j + 1
    i = i + 1

  # get bpundaries
  boundaries = []
  for i in range(text_len):
    boundaries.append(float(0))
  
  i = text_len
  while i > 0:
    boundaries[i-1] = 1
    try:
      prev[i-1]
      i = prev[i-1]
    except IndexError:
      i = i - 1

  # fill result
  result = []
  prev = 0
  for i in range(len(boundaries)):
    if boundaries[i] != float(0):
      wd_start = prev
      wd_end = i + 1
      current_wd = x[wd_start:wd_end]
      result.append(current_wd)
      prev = wd_end

  if character_base:
    result_char = [ i for i in x]
    if result_char != result:
      return (result, result_char)
    else:
      return (result, ['<NON>']) # avoid shape not equal.
  else:
    return (result, ['<NON>']) # avoid shape not equal.


class ChineseSegmentor2():
  '''
      as same as Jieba's algorithm
  '''
  def __init__(self, lex):
    self.lex = lex
    

  def DAG(self, x):
    '''
      台灣大學生
      DAG = {0:[0, 1, 2], 1:[1], 2:[2, 3, 4], 3:[3, 4], 4:[4]}
    '''
    x = str(x)
    _len = len(x)
    _dag = {}
    for i in range(_len):
      for j in range(i, _len):
        if i == j:
          try:
            _ = self.lex[x[i]]
            try:
              _dag[i].append(j)
            except KeyError:
                _dag[i] = [j]
          except KeyError:
            continue

        else:
          try:
            _ = self.lex[x[i:j+1]]
            try:
              _dag[i].append(int(j))
            except KeyError:
              _dag[i] = [int(j)]
          except KeyError:
            continue
        
    return _dag

  def segment(self, x):
    _route = {}
    _dag = self.DAG(x)
    _len = len(x)
    _route[_len] = (0, 0)
    _logtotal = math.log(0.5)
    for idx in range(_len - 1, -1, -1):
      temp = []
      for idy in _dag[idx]:
        temp.append((math.log(self.lex[x[idx:idy + 1]] - _logtotal + _route[idy + 1][0]), idy))
        _route[idx] = max(temp)
        
    i = 0
    buf = ''
    re_eng = re.compile('[a-zA-Z0-9]', re.U)
    while i < _len:
      j = _route[i][1] + 1
      l_word = x[i:j]
      if re_eng.match(l_word) and len(l_word) == 1:
        buf += l_word
        i = j
      else:
        if buf:
          yield buf
          buf = ''
        yield l_word
        i = j
      if buf:
        yield buf
        buf = ''


def make_context_fst(x, write=True):
  '''
    read a Kaldi lexicon format text.
      <word1> <weight> <sub-word1> <sub-word2> <...>
      example:
        ABABA 1.0 ABABA
        ABACHA 1.0 ABACHA
        每日一物 100 每 日 一 物
        每日一物 100 每日 一物
    Returns:
      List with FST format or write file.
  '''
  C = read_file_to_list(x)
  C_fst = []
  state = int(0)
  for i in range(len(C)):
    if len(C[i]) == 3:
      logprob = '%.10f' % (-math.log(float(C[i][1])))
      C_fst.append(['0', '0', C[i][2], C[i][0], logprob])
    else:
      logprob = '%.10f' % (-math.log(float(C[i][1])))
      for j in range(len(C[i]) - 2):
        if j == 0:
          C_fst.append(['0', '%s' %  (state + 1), C[i][j+2], C[i][0], logprob])
          state = state + 1
        elif j == len(C[i]) - 3:
          C_fst.append(['%s' % state, '0', C[i][j+2], '<eps>'])
        else:
          C_fst.append(['%s' % state, '%s' % (state + 1), C[i][j+2], '<eps>'])
          state = state + 1
  C_fst.append(['0','0']) # add end
  
  if write:
    with io.open(x + '.fst', 'a+', encoding='utf8') as f:
      for i in range(len(C_fst)):
        f.write('\t'.join(C_fst[i]) + '\n')
  else:
      return(C_fst)


def test():
    lex = {'台灣':0.1 , '大':0.3 , '學生': 0.3 , '大學生': 0.4, '台': 0.05, '灣': 0.05, '學': 0.1, '生':0.1, '台灣大': 0.08, '大學': 0.2}
    a = ChineseSegmentor2(lex)
    print(a.DAG('台灣大學生'))
    print(' '.join(a.segment('台灣大學生')))


if __name__ == '__main__':
    test()