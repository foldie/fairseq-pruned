#!/bin/bash

function die {
  echo "$1" >&2
  exit 1
}

function is_checkpoint_id {
	echo "checking $1"
	if [[ $1 =~ ^[0-9_]+$ ]]
	then
		echo "valid checkpoint id"
    	true
	else
		echo "not a valid checkpoint id"
    	false
	fi
}

if [ "$#" -ne 3 ] || ! [ -d "$1" ]; then
	echo "USAGE: $0 <path-to-checkpoints> $1 <language> $2 <corpus_size> $3"
	exit 1
fi
cp_path="$1"
lang="$2"
size="$3"

for checkpoint in $cp_path/checkpoint*.pt; do
	checkpoint_id=$(echo "$checkpoint" | sed -E -e 's/^.*checkpoint([0-9_]+)\.pt$/\1/')
	echo "$checkpoint: $checkpoint_id"
	if is_checkpoint_id "$checkpoint_id"; then
		if [[ "$checkpoint" =~ last || "$checkpoint" =~ best ]]; then
			echo "Not evaluating best/last checkpoint: $checkpoint"
		else
			[ -d "$cp_path/bleus" ] || mkdir $cp_path/bleus
			output_file="$cp_path/bleus/bleu${checkpoint_id}"
			echo "found valid checkpoint $checkpoint"
			PYTHONPATH=$(pwd) python3 fairseq_cli/generate.py "/home/users2/foeldeni/fs/data-bin/wmt18.tokenized.en-${lang}_${size}" --path "$checkpoint"  --source-lang en --target-lang $lang --beam 5 --batch-size 4 --remove-bpe > "$output_file" && rm $checkpoint
		fi
	fi
done
