#!/bin/bash

if [ "$#" -ne 1 ] || ! [ -d "$1" ]; then
	echo "USAGE: $0 <path-to-checkpoint-files>"
	exit 1
fi

sorted_files=$(ls $1/bleus | $(dirname $0)/sort_bleus.py )
while read -r file ; do
	grep 'BLEU4' "$1/bleus/$file" | sed -r -e 's/^.*BLEU4 = ([0-9.]*).*$/\1/'
done <<< "$sorted_files"
