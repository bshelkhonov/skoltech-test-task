#!/bin/bash

data=$((tr -d '\015' | tr -s ' ' | tr ' ' '\n' | grep . | sort | uniq -c) < $1)
data=$(sort -r -nk1 <<< "$data")

iter=0

mkdir -p "$2"

echo "$data" | while read line
do
    filename="$"$(awk '{print $2"_"$1}' <<< $line)
    touch "$2""/""$filename"
    ((iter += 1))
    if ((iter == 10)); then
        break
    fi
done