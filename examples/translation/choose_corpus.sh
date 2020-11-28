#!/usr/bin/env bash
#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh
# usage: language, max lines, max test lines


set -euo pipefail
set -a

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git || true

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git || true

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=40000
START_DIR=$(realpath $(dirname "$0"))
src=en

#get a target language as input or default to german. available languages: en, cs, hu
if [[ $# -ge 1 ]]; then
    tgt=$1
else
    tgt='de'
fi

#get a maximum number of lines for the training set: currently it's being automatically set to 2.5 million sentences
if [[ $# -ge 2 ]]; then
    max_lines=$2
else
    max_lines=-1
fi

#get a maximum number of lines for the test sentences: at the moment this is 3000 sentences
if [[ $# -ge 3 ]]; then
    max_test_lines=$3
else
    max_test_lines=-1
fi

#spacy tokenizing for hungarian and turkish, moses tokenizing for everything else
#normalises punctuation and removes non printable characters too
function tokenize {
  case $2 in
  de)
      cat $1 | \
      perl $NORM_PUNC -l $2 | \
      perl $REM_NON_PRINT_CHAR | \
      perl $TOKENIZER -threads 8 -a -l $2 > $3
      ;;
  en)
      cat $1 | \
      perl $NORM_PUNC -l $2 | \
      perl $REM_NON_PRINT_CHAR | \
      perl $TOKENIZER -threads 8 -a -l $2 > $3
      ;;
  cs)
      cat $1 | \
      perl $NORM_PUNC -l $2 | \
      perl $REM_NON_PRINT_CHAR | \
      perl $TOKENIZER -threads 8 -a -l $2 > $3
      ;;
  hu)
      cat $1 | \
      perl $NORM_PUNC -l $2 | \
      perl $REM_NON_PRINT_CHAR | \
      python3 $START_DIR/tokenize_hungarian.py > $3
      ;;
  tr)
      cat $1 | \
      perl $NORM_PUNC -l $2 | \
      perl $REM_NON_PRINT_CHAR | \
      python3 $START_DIR/tokenize_turkish.py > $3
      ;;
  *)
    echo "unknown language passed to tokenizer"
    exit 1
    ;;
  esac
}

# a function for populating the URLS, FILES, CORPORA and UNPACKED arrays to contain the language specific urls and files
case $tgt in
  de)
    function init_arrays {
      URLS=(
          "https://s3.amazonaws.com/web-language-models/paracrawl/release1/paracrawl-release1.en-de.zipporah0-dedup-clean.tgz"
          "http://data.statmt.org/wmt18/translation-task/dev.tgz"
          "http://data.statmt.org/wmt18/translation-task/test.tgz"
      )
      FILES=(
          "paracrawl-release1.en-de.zipporah0-dedup-clean.tgz"
          "dev.tgz"
          "test.tgz"
      )
      CORPORA=(
          "paracrawl-release1.en-de.zipporah0-dedup-clean"
      )
      UNPACKED=(
          "paracrawl-release1.en-de.zipporah0-dedup-clean.en"
          "dev"
          "test"
      )
    }
  ;;
  cs)
    function init_arrays {
      URLS=(
          "https://s3.amazonaws.com/web-language-models/paracrawl/release1/paracrawl-release1.en-cs.zipporah0-dedup-clean.tgz"
          "http://data.statmt.org/wmt18/translation-task/dev.tgz"
          "http://data.statmt.org/wmt18/translation-task/test.tgz"
      )
      FILES=(
          "paracrawl-release1.en-cs.zipporah0-dedup-clean.tgz"
          "dev.tgz"
          "test.tgz"
      )
      CORPORA=(
          "paracrawl-release1.en-cs.zipporah0-dedup-clean"
      )
      UNPACKED=(
          "paracrawl-release1.en-cs.zipporah0-dedup-clean.en"
          "dev"
          "test"
      )
    }
  ;;
  hu)
    function init_arrays {
      URLS=(
          "https://s3.amazonaws.com/web-language-models/paracrawl/release9/en-hu/en-hu.txt.gz"
          "http://data.statmt.org/wmt18/translation-task/dev.tgz"
          "https://www.statmt.org/wmt09/test.tgz"
      )
      FILES=(
          "en-hu.txt.gz"
          "dev.tgz"
          "test.tgz"
      )
      CORPORA=(
          "en-hu.txt"
      )
      UNPACKED=(
          "en-hu.txt"
          "dev"
          "test"
      )
    }
  ;;
  tr)
    function init_arrays {
      URLS=(
          "https://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/moses/en-tr.txt.zip"
          "http://data.statmt.org/wmt18/translation-task/dev.tgz"
          "http://data.statmt.org/wmt18/translation-task/test.tgz"
      )
      FILES=(
          'download.php?f=OpenSubtitles%2Fv2018%2Fmoses%2Fen-tr.txt.zip'
          "dev.tgz"
          "test.tgz"
      )
      CORPORA=(
          'OpenSubtitles.en-tr'
      )
      UNPACKED=(
          'OpenSubtitles.en-tr.en'
          "dev"
          "test"
      )
    }
  ;;
esac

export -f tokenize init_arrays
$(dirname $0)/prep.sh