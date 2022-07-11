#!/bin/sh

((tr -d '\015' | tr -s ' ' | tr ' ' '\n' | grep . | sort | uniq -c | awk {'print $2" "$1'}) < $1)
