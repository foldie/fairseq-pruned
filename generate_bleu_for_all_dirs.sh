for corp_size in 1500000; do
    for lang in de cs hu; do
        if [ -d /home/users2/foeldeni/arbeitsdaten/checkpoints/$lang/$corp_size/$1 ]; then
            ./generate_bleu_for_checkpoint.sh /home/users2/foeldeni/arbeitsdaten/checkpoints/$lang/$corp_size/$1 $lang $corp_size
        fi
    done
done

#for corp size in 1500000 1750000 2000000 2250000 2500000 500000 750000; do