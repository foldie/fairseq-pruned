#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh
set -euo pipefail

#function to call in case an error occurs and we would like to exit
function die {
  echo "error: $1" >&2
  exit 1
}

#helper function to get random seed and shuffle two files in exactly the same way
function get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}


OUTDIR=wmt18_${src}_${tgt}_$max_lines
lang=$src-$tgt
prep=$OUTDIR
tmp=$prep/tmp
orig=orig/$tgt
shortened_original_corpus=orig/$tgt/$max_lines
dev=dev/newstest2013

#get arrays from choose corpus where language selection happens
#(URLS, FILES, CORPORA and UNPACKED gets set in choose_corpora.sh)
init_arrays

#check if moses exists at all at the right path
if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

mkdir -p $orig $tmp $prep $shortened_original_corpus

cd $orig

#downloading files in case they don't yet exist in the specified location and decompressing them
echo "Checking if corpus files are already there..."
for ((i=0;i<${#URLS[@]};++i)); do
    file=${FILES[i]}
    unp=${UNPACKED[i]}
    if [ -f $file ]; then
        echo "$file already exists, skipping download"
    elif [ -d $unp ] || [ -f $unp ]; then
        echo "$unp already exists, skipping download"
    else
        url=${URLS[i]}
        wget "$url"
        if [ -f "$file" ]; then
            echo "$url successfully downloaded."
        else
            echo "$url not successfully downloaded."
            exit 1
        fi
        if [ ${file: -4} == ".tgz" ]; then
            tar zxvf $file
            rm $file
        elif [ ${file: -4} == ".tar" ]; then
            tar xvf $file
            rm $file
        elif [ ${file: -3} == ".gz" ]; then
            gzip -d $file
        elif [ ${file: -4} == ".zip" ]; then
            unzip $file
            rm $file
        fi
        echo "Successfully unzipped $file."
    fi
done

cd ..
pwd
echo "pre-processing train data..."

#split the training files if they aren't yet split into target and source language files, one sentence per line
#prepares the files based on their extensions, like whether they are tab separated or in an xml format
for f in "${CORPORA[@]}"; do
    if [ ! -f "$tgt/$f.$tgt" ] ||  [ ! -f "$tgt/$f.$src" ]; then
        if [ -f $tgt/$f ] && [ ${f: -4} == ".txt" ]; then
            num_sents=$(wc -l "$tgt/$f")
            python3 $START_DIR/prep_tab.py "$tgt/$f" "$tgt/$f.$src" "$tgt/$f.$tgt"
        elif [ -f $tgt/$f ] && [ ${f: -4} == ".tmx" ]; then
            num_sents=$(python3 $START_DIR/validate_tmx.py "$tgt/$f")
            python3 $START_DIR/prep_tmx.py "$tgt/$f" "$tgt/$f.$src" "$tgt/$f.$tgt"
        elif [ -d $tgt/$f ]; then
          [ -f "$tgt/$f.$tgt" ] ||  die "file is missing"
          [ -f "$tgt/$f.$src" ] ||  die "file is missing"
          num_sents=$(wc -l "$tgt/$f")
        else
          echo "$f"
          echo "$tgt/$f"
          die "unexpected input to pre-processing step"
        fi
    fi
    #we assume both source and target files to have exactly the same lengths
    [ $(wc -l < "$tgt/$f.$tgt") == $(wc -l < "$tgt/$f.$src") ] || die "different number of sentences in source and target files"
done
cd ..

echo "Shortening the corpus files to maximum number of lines: $max_lines ..."
#shorten the training files into the maximum number of lines defined in choose_corpus.sh by the variable: max_lines
#at the moment the training files will be shortened into 2.5mil sentences
for l in $src $tgt; do
    #rm -f $tmp/train.tags.$lang.tok.$l
    for f in "${CORPORA[@]}"; do
        num_lines=$(wc -l < "$orig/$f.$l")
        if [[ $max_lines -lt 0 ]]; then
            filtered=$num_lines
            ln -sf $(realpath "$orig/$f.$l") "$shortened_original_corpus/$f.$l.short"
        else
            filtered=$max_lines
            head -n "$filtered" "$orig/$f.$l" > "$shortened_original_corpus/$f.$l.short"
        fi
        #filtered=$(echo "($num_lines * $percentage)/1" | bc)
        if [ ! -f "$tmp/train.tags.$lang.tok.$l" ]; then
          tokenize "$shortened_original_corpus/$f.$l.short" "$l" "$tmp/train.tags.$lang.tok.$l"
        fi
    done
done

for f in "${CORPORA[@]}"; do
  #we assume both source and target files to continue having exactly the same lengths relative to each other
  #after shortening and tokenizing
  [ $(wc -l < "$tmp/train.tags.$lang.tok.$tgt") == $(wc -l < "$tmp/train.tags.$lang.tok.$src") ] || die "different number of sentences in $tmp/train.tags.$lang.tok.$src and $tmp/train.tags.$lang.tok.$tgt files"
done

#preprocessing test files from the news corpus 2018 has test files for all languages except hungarian
#test files then come from the 2009 data
echo "pre-processing test data..."
for l in $src $tgt; do
    if [ "$l" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi
    case $tgt in
      hu)
          testfile=$orig/test/newstest2009-src.$l.sgm
          ;;
      *)
          testfile=$orig/test/newstest2018-$src$tgt-$t.$l.sgm
          ;;
    esac
    #shorten the test files into the maximum number of lines defined in choose_corpus.sh by the variable: max_test_lines
    #at the moment the testing files will be shortened into 3000 sentences
    if [ -f "$testfile.short" ]; then
        echo "$testfile.short already exists, skipping the shortening step"
    else
        if [[ $max_test_lines -lt 0 ]]; then
            filtered=$num_lines
            ln -sf $(realpath "$testfile") "$testfile.short"
        else
            filtered=$max_test_lines
            head -n "$filtered" "$testfile" > "$testfile.short"
        fi
    fi
    #preprocesses sgm files to strip them from xml tags
    grep '<seg id' $testfile.short | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" > $tmp/test_temp.$l
        tokenize $tmp/test_temp.$l $l $tmp/test.$l
    echo ""
done

#dev is a specific percentage of the train file, here it's mathed so that it will shorten to 1500 sentences
echo "splitting train and valid..."
for l in $src $tgt; do
    all_lines=$(wc -l $tmp/train.tags.$lang.tok.$l)
    cut=$(echo $all_lines | perl -nl -MPOSIX -e 'print floor($_ / 1500);')
    if [[ $cut == 0 ]]; then
        echo "Warning: Training corpus length should ideally be longer than 1500 lines. 
        All training lines will be used in the validation set.
        Furthermore learning BPE is not guaranteed to succeed on particularly short files "
        cut=1
    fi
    awk "{if (NR % $cut == 0)  print \$0; }" $tmp/train.tags.$lang.tok.$l > $tmp/valid.$l
    awk "{if (NR % $cut != 0)  print \$0; }" $tmp/train.tags.$lang.tok.$l > $tmp/train.$l
done

#learning bpe and applying bpe, mainly untouched original script from fairseq/examples/translation/prepare-wmt14en2de.sh
TRAIN=$tmp/train.$src-$tgt
BPE_CODE=$prep/code
rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
done

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $tmp/bpe.$f
    done
done

#we assume both source and target files to continue having exactly the same lengths relative to each other
#after getting bpe applied to them
for f in train valid test; do
  [ $(wc -l < "$tmp/bpe.$f.$tgt") == $(wc -l < "$tmp/bpe.$f.$src") ] || die "different number of sentences in source and target files"
done

#cleaning pearl script, mainly untouched original script from fairseq/examples/translation/prepare-wmt14en2de.sh
#additional regex substitutions and getting rid of lines where the number of words are vastly different (ratio)
perl $CLEAN -ratio 1.5 $tmp/bpe.train $src $tgt $prep/train 1 250
perl $CLEAN -ratio 1.5 $tmp/bpe.valid $src $tgt $prep/valid 1 250

for L in $src $tgt; do
    cp $tmp/bpe.test.$L $prep/test.$L
done

# train valid and test all get shuffled here
for data in train valid test; do
    random_number=$RANDOM
    echo $random_number > $prep/$data.seed
    for L in $src $tgt; do
        shuf --random-source=<(get_seeded_random $random_number) -o $prep/$data.$L < $prep/$data.$L
    done
done

#remove the tmp directory to free up space
rm -r $tmp