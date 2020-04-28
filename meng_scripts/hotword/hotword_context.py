import os
import io
import argparse
import math

non_hotwords_weight = 0.1

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

def check_oov_and_merge(words, words_table, list=False):
  '''
    check oov and merge oov into words.txt.
    Returns:
        dict or list
  '''
  words_table = read_file_to_dict(words_table)
  words = read_file_to_list(words)
  count = len(words_table) - 1
  for i in words:
    try:
      # if word in word_table don't need to add new word id.
      key = ''.join(i)
      _ = words_table[key]
    except KeyError:
      count = count + 1
      dic = {key: str(count)}
      words_table.update(dic)

  if list:
    return [ [k,v] for k, v in words_table.items() ]
  else:
    return words_table

def check_char_illegal(x, words_table, write=False):
  '''
    read a file and check each character.
    Returns:
        legal result with list format or write in the text with ".cleanup" flac.
  '''
  out = []
  text = read_file_to_list(x)
  words_table = read_file_to_dict(words_table)
  for i in range(len(text)):
    flac = 'True'
    for j in [a for a in ''.join(text[i])]:
      try:
        words_table[j]
      except KeyError:
        print('include non legel character')
        print('remove line :' + ' ' + ' '.join(text[i]))
        flac = 'False'
    if flac != 'False':
      out.append(text[i])
  
  if write:
    with io.open(x + '.cleanup', 'a+', encoding='utf-8') as f:
      for i in out:
        f.write(''.join(i) + '\n')
  else:
    return out  

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

def make_context_fst(x, write=True, ):
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

def main(args):
  os.makedirs(args.output_dir)
  # stage 1. check hotword table inculde oov / illegal char or not.
  # if include oov, list in the oov table.
  # if contain illegial char, remove the line.
  check_char_illegal(args.hotwords, args.words, True)
  hotwords_table = read_file_to_list(args.hotwords + '.cleanup') 
  new_words_table = check_oov_and_merge(args.hotwords + '.cleanup', args.words)
  with io.open(args.output_dir + '/words.txt', 'a+', encoding='utf-8') as f:
    for key, value in new_words_table.items():
      f.write(key + ' ' + value + '\n')

  # stage 2. setup context fst.
  with io.open(args.output_dir + '/C.txt', 'a+', encoding='utf-8') as out:
    for i in read_file_to_list(args.words):
      if i[0] != '<eps>':
        if [i[0]] not in hotwords_table:
          out.write(i[0] + '\t' + str(non_hotwords_weight) + '\t' + i[0] + '\n')

  # stage 3. re-segment hotword & oov word into word-piece units.
  # note: here involved character and pre-LM unigram based representation.
  with io.open(args.output_dir + '/C.txt', 'a+', encoding='utf-8') as out:
    for i in range(len(hotwords_table)):
      for j in ChineseSegmenter(''.join(hotwords_table[i]), args.unigram, True):
        if j != ['<NON>']:
          out.write(''.join(hotwords_table[i]) + '\t' + str(args.weight) + '\t' + ' '.join(j) + '\n') 

  # stage 4. build and compile C.txt to C.fst with OpenFST tool.
  make_context_fst(args.output_dir + '/C.txt')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='This command create a Context FST for using in HCLGC.fst composed')
  parser.add_argument('hotwords', help='hotwords list')
  parser.add_argument('words', help='words.txt')
  parser.add_argument('unigram', help='pre-LM unigram probability')
  parser.add_argument('weight', help='hotwords weight, ex:100')
  parser.add_argument('output_dir', help='storege dir')
  args = parser.parse_args()
  main(args)
