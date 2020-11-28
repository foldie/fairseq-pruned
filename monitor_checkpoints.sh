#!/usr/bin/env bash
set -euo pipefail
while true; do
    for corp_size in 1500000; do
        for lang in de cs hu; do
            chdir="/home/users2/foeldeni/arbeitsdaten/checkpoints/$lang/$corp_size/"
            for dirname in $chdir/*; do
                if [[ $dirname != *bleus ]] && [[ $dirname != *logs ]] && [[ $dirname != *logs ]]; then
                    num_checkpoints=$(ls $dirname | wc -l)
                    if [ $num_checkpoints -gt 5 ]; then
                        chend=$(grep -o "pr*bl*fs*_[0-3][0-9]_[0-1][0-9]_20[0-9][0-9]@[0-2][0-9]:[0-5][0-9]" <<< $dirname)
                        ./gen_bleu_for_all_dirs.sh $chend
                    fi
                fi
            done
        done
    done
    sleep 30m
done
