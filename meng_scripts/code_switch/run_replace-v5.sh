#!/bin/bash

## replace function has been moved into the while loop
## rule replace in each $line

table=$2
#count=1

while read line; do
#  echo $count
  flag=1
  word=`echo $line | cut -f 1 -d " "`
  phone=`echo $line | cut -f 2- -d " "`
## semi-vowel "L"
  if [[ $phone =~ "AH0 L " || $phone =~ "AH0 L"$ ]]; then
    ## first, check L in the end or not
    if echo $word | grep -q "L$"; then
      base_word=$word
      flag=2
      replace=`echo $phone | sed 's/AH0 L/o u/g'`
      line="$word $replace"
      word=`echo $line | cut -f 1 -d " "`
      phone=`echo $line | cut -f 2- -d " "`
    
    fi
    ## second, check pronunciation rule
    if [[ $word =~ "AL" || $word =~ "EL" || $word =~ "IL" || $word =~ "OL" || $word =~ "UL" ]]; then
      if [[ $word =~ $base_word && $flag == 1 ]]; then
        replace=`echo $phone | sed 's/AH0 L/o u/g'`
        line="$word $replace"
        word=`echo $line | cut -f 1 -d " "`
        phone=`echo $line | cut -f 2- -d " "`

      fi
    else
      replace=`echo $phone | sed 's/AH0 L/o u/g'`
      line="$word $replace "
      word=`echo $line | cut -f 1 -d " "`
      phone=`echo $line | cut -f 2- -d " "`

    fi
  fi
## semi-vowel "N"
  if [[ $phone =~ "AH0 N " || $phone =~ "AH0 N"$ ]]; then
    ## first, check N in the end or not
    if echo $word | grep -q "N$"; then
      base_word=$word
      falg=2
      replace=`echo $phone | sed 's/AH0 N$/ern/g'`
      line="$word $replace"
      word=`echo $line | cut -f 1 -d " "`
      phone=`echo $line | cut -f 2- -d " "`

    fi
    ## second, check pronunciation rule
    if [[ $word =~ "AN" || $word =~ "EN" || $word =~ "IN" || $word =~ "ON" || $word =~ "UN" ]]; then
      if [[ $word =~ $base_word && $flag == 1 ]]; then
        replace=`echo $phone | sed 's/AH0 N /ern /g'`
        line="$word $replace"
        word=`echo $line | cut -f 1 -d " "`
        phone=`echo $line | cut -f 2- -d " "`

      fi
      ## special rule. ex: UNFORTUNATELY
      if [[ $word =~ ^UN && $phone =~ ^AH0" N" ]]; then
         replace=`echo $phone | sed 's/^AH0 N /a ng /g'`
         line="$word $replace"
         word=`echo $line | cut -f 1 -d " "`
         phone=`echo $line | cut -f 2- -d " "`

      fi
    else
      replace=`echo $phone | sed 's/AH0 N /ern /g'`
      line="$word $replace"
      word=`echo $line | cut -f 1 -d " "`
      phone=`echo $line | cut -f 2- -d " "`

    fi

  fi
## semi-vowel "M"
  if [[ $phone =~ "AH0 M " || $phone =~ "AH0 M"$ ]]; then
    ## first, check M in the end or not
    if echo $word | grep -q "M$"; then
      base_word=$word
      flag=2
      replace=`echo $phone | sed 's/AH0 M/ern/g'`
      line="$word $replace"
      word=`echo $line | cut -f 1 -d " "`
      phone=`echo $line | cut -f 2- -d " "`

    fi
    ## second, check pronunciation rule
    if [[ $word =~ "AM" || $word =~ "EM" || $word =~ "IM" || $word =~ "OM" || $word =~ "UM" ]]; then
      if [[ $word =~ $base_word ]]; then
        replace=`echo $phone | sed 's/AH0 M/ern/g'`
        line="$word $replace"
        word=`echo $line | cut -f 1 -d " "`
        phone=`echo $line | cut -f 2- -d " "`

      fi
      ## special rule um, ex:UMBRELLA
      if [[ $word =~ "UM" && $phone =~ ^AH0" M" ]]; then
        replace=`echo $phone | sed 's/^AH0 M/a ng/g'`
        line="$word $replace"
        word=`echo $line | cut -f 1 -d " "`
        phone=`echo $line | cut -f 2- -d " "`

      fi
    else
      replace=`echo $phone | sed 's/AH0 M/ern/g'`
      line="$word $replace"
      word=`echo $line | cut -f 1 -d " "`
      phone=`echo $line | cut -f 2- -d " "`

    fi

  fi

  echo $line | apply_map.pl -f 2- $table >> new_lex.txt || exit 1;
#  count=$((count + 1))
done < $1
sort -u new_lex.txt > lexicon.txt
