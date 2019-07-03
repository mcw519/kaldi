#!/bin/bash

## replace function has been moved into the while loop
## rule replace in each $line

table=$2
count=1

while read line; do
  echo $count
  word=`echo $line | cut -f 1 -d " "`
  phone=`echo $line | cut -f 2- -d " "`
## semi-vowel "L"
  if [[ $phone =~ "AH0 L " || $phone =~ "AH0 L"$ ]]; then
    ## first, check L in the end or not
    if echo $word | grep -q "L$"; then
      base_word=$word
      replace=`echo $phone | sed 's/AH0 L/o u/g'`
      line="$word $replace"
    ## second, check pronunciation rule
    elif [[ $word =~ "AL" || $word =~ "EL" || $word =~ "IL" || $word =~ "OL" || $word =~ "UL" ]]; then
      if [[ $word =~ $base_word ]]; then
        replace=`echo $phone | sed 's/AH0 L/o u/g'`
        line="$word $replace"

      fi
    ## special rule ul, ex:ULTIMATE, ULTRA
    elif [[ $word =~ "UL" ]]; then
      if [[ $word =~ ^UL ]]; then
        replace=`echo $phone | sed 's/^AH0 L/a u/g'`
        line="$word $replace"

      fi
    else
      replace=`echo $phone | sed 's/AH0 L/o u/g'`
      line="$word $replace "

    fi

  fi
## semi-vowel "N"
  if [[ $phone =~ "AH0 N " || $phone =~ "AH0 N"$ ]]; then
    ## first, check N in the end or not
    if echo $word | grep -q "N$"; then
      base_word=$word
      replace=`echo $phone | sed 's/AH0 N/ern/g'`
      line="$word $replace"
    ## second, check pronunciation rule
    elif [[ $word =~ "AN" || $word =~ "EN" || $word =~ "IN" || $word =~ "ON" ]]; then
      if [[ $word =~ $base_word ]]; then
        replace=`echo $phone | sed 's/AH0 N/ern/g'`
        line="$word $replace"

      fi
    ## special rule un
    elif [[ $word =~ "UN" ]]; then
      if [[ $word =~ ^UN ]]; then
        replace=`echo $phone | sed 's/^AH0 N/a ng/g'`
        line="$word $replace"

      fi
    else
      replace=`echo $phone | sed 's/AH0 N/ern/g'`
      line="$word $replace"

    fi

  fi
## semi-vowel "M"
  if [[ $phone =~ "AH0 M " || $phone =~ "AH0 M"$ ]]; then
    ## first, check M in the end or not
    if echo $word | grep -q "M$"; then
      base_word=$word
      replace=`echo $phone | sed 's/AH0 M/ern/g'`
      line="$word $replace"
    ## second, check pronunciation rule
    elif [[ $word =~ "AM" || $word =~ "EM" || $word =~ "IM" || $word =~ "OM" ]]; then
      if [[ $word =~ $base_word ]]; then
        replace=`echo $phone | sed 's/AH0 M/ern/g'`
        line="$word $replace"

      fi
    ## special rule um, ex:UMBRELLA
    elif [[ $word =~ "UM" ]]; then
      if [[ $word =~ ^UM ]]; then
        replace=`echo $phone | sed 's/^AH0 M/a ng/g'`
        line="$word $replace"

      fi
    else
      replace=`echo $phone | sed 's/AH0 M/ern/g'`
      line="$word $replace"

    fi

  fi

  echo $line | apply_map.pl -f 2- $table >> new_lex.txt || exit 1;
  count=$((count + 1))
done < $1
sort -u new_lex.txt > lexicon.txt
