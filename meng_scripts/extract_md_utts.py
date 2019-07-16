#-*- coding:utf-8 -*-
import io
import sys
import re


with io.open(sys.argv[1], 'r' , encoding = 'utf-8') as f:
  content = f.readlines()

for i in range(0, len(content)):
  out = io.open(sys.argv[2], 'a+', encoding = 'utf-8' )
  line = content[i]
  zhPattern = re.compile(u'[\u4e00-\u9fa5]+')
  match = zhPattern.search(content[i])
  if match:
    out.write(line)
