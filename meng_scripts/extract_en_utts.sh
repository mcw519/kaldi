#!/bin/bash

while read file; do
  text=`echo $file | cut -d' ' -f2-`;
  id=`echo $file | cut -d' ' -f1`;
  if [[ "$text" =~ [A-Za-z] ]]; then 
    echo "$id $text";
  else
    continue;
  fi
done < $1
