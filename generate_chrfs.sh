#!/usr/bin/env bash
set -euo pipefail
for corp_size in $1; do
    for lang in hu de cs; do
        # path=$HOME/arbeitsdaten/checkpoints/$lang/$corp_size/b_19_1/
        for path in "$HOME/checkpoints/$lang/$corp_size/"*; do
            if [ -d "$path/bleus" ] && [ ! -d "$path/chrfs" ]; then
                echo "Generating chrfs scores for $path"
                mkdir "$path/chrfs"
                for filename in "$path"/bleus/*; do
                    echo "$filename"
                    newfilename=$(sed "s/bleu/chrf/g" <<< $filename)
                    awk 'BEGIN {FS="\t"}; { if ($1 ~ /^H-/) print $3}' "$filename" > "$newfilename.hyp"
                    awk 'BEGIN {FS="\t"}; { if ($1 ~ /^T-/) print $2}' "$filename" > "$newfilename.ref"
                    hyps=$(wc -l < "$newfilename.hyp")
                    refs=$(wc -l < "$newfilename.ref")
                    if [ "$hyps" -gt 0 ] && [ "$refs" -gt 0 ]; then
                        python3 "$HOME/fs/chrF/chrF++.py" -H "$newfilename".hyp -R "$newfilename".ref > "$newfilename".chrf
                    fi
                done
            else
                if [ ! -d "$path/bleus" ]; then
                    echo "Skipping path $path because $path/bleus doesn't exist"
                else 
                    echo "Skipping path $path because $path/chrfs already exists"
                fi
            fi
        done
    done
done
