#! /usr/bin/enve/ python3

# Copyright 2020 (author: Meng Wu)

import os
import io
import argparse
import math
from smart_reading import smart_read
from common import check_oov_and_merge, read_file_to_dict, read_file_to_list, ChineseSegmenter2, make_context_fst

non_hotwords_weight = 0.1

def main(args):
  os.makedirs(args.output_dir)
  wd_table = read_file_to_dict(args.words)

  hot_wd_table = []
  f = io.open(args.hotwords, 'r')
  c_out = io.open(args.output_dir + '/C.txt', 'a+', encoding='utf-8')

  for i in f.readlines():
    # check hotword table inculde oov/illegal char or not.
    # if contain illegial char, remove the line.
    _type, _content = smart_read([i], wd_table)

    if _content == False:
      continue

    if _type == 3:
      hot_wd = _content[0]
    else:
      hot_wd = _content
    
    hot_wd_table.append(hot_wd)
    wd_table = check_oov_and_merge(hot_wd, wd_table)
    
    if _type == 1:
      tokenizor = ChineseSegmenter2(args.unigram)
      result = list(tokenizor.segment(hot_wd))
      c_out.write(hot_wd + '\t' + str(args.weight) + '\t' + ' '.join(result) + '\n')

    elif _type == 2:
      c_out.write(hot_wd + '\t' + str(args.weight) + '\t' + hot_wd + '\n')

    elif _type == 3:
      c_out.write(hot_wd + '\t' + str(args.weight) + '\t' + ' '.join(_content[1:]) + '\n')
  
  f.close()
  c_out.close()

  with io.open(args.output_dir + '/C.txt', 'a+', encoding='utf-8') as c_out:
    for i in read_file_to_list(args.words):
      if i[0] != '<eps>':
        if i[0] not in hot_wd_table:
          c_out.write(i[0] + '\t' + str(non_hotwords_weight) + '\t' + i[0] + '\n')

  # create new words.txt include new hotwords.
  with io.open(args.output_dir + '/words.txt', 'a+', encoding='utf-8') as f:
    for key, value in wd_table.items():
      f.write(key + ' ' + value + '\n')
  
  # build and compile C.txt to C.fst with OpenFST tool.
  make_context_fst(args.output_dir + '/C.txt', write=True)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='This command create a Context FST for using in HCLGC.fst composed')
  parser.add_argument('hotwords', help='hotwords list')
  parser.add_argument('words', help='words.txt')
  parser.add_argument('unigram', help='pre-LM unigram probability')
  parser.add_argument('weight', help='hotwords weight, ex:100')
  parser.add_argument('output_dir', help='storege dir')
  args = parser.parse_args()
  main(args)
