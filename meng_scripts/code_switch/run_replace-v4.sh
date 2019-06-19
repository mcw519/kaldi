#!/bin/bash

table=$2

while read line; do
  word=`echo $line | cut -f 1 -d " "`
  phone=`echo $line | cut -f 2- -d " "`

  if [[ $phone =~ "AH0 L" ]]; then
    ## first, check L in the end or not
    if echo $word | grep -q "L$"; then
       base_word=$word
       replace=`echo $phone | sed 's/AH0 L/o u/g'`
       line="$word $replace"
       echo $line | apply_map.pl -f 2- $table >> new_lex.txt
    ## second, check pronunciation rule
    elif [[ $word =~ "AL" || $word =~ "EL" || $word =~ "IL" || $word =~ "OL" ]]; then
       if [[ $word =~ $base_word ]]; then
         replace=`echo $phone | sed 's/AH0 L/o u/g'`
         line="$word $replace"
       else
         echo $line | apply_map.pl -f 2- $table >> new_lex.txt

       fi
    else
       replace=`echo $phone | sed 's/AH0 L/o u/g'`
       line="$word $replace"

      echo $line | apply_map.pl -f 2- $table >> new_lex.txt

    fi

  else
    echo $line | apply_map.pl -f 2- $table >> new_lex.txt

  fi

done < $1
