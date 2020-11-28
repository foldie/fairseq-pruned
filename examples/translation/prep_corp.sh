#!/usr/bin/env bash
set -euo pipefail

#run preparation script with different max lines parameters producing separate directories of data
function print_usage {
  cat << EndOfMessage
$(basename "$0") [-h] [-t TARGET_LANGUAGE] [-m MAX_LINES] [-j DECREMENT] -- given a target language, corpus size and decrement
prepare all the corpora and run training.
where:\n
    -h  show this help text
    -t  set the target language (en, hu, de, all)
    -m  set the maximum number of lines
    -j  set the number decrement lines per experiment
EndOfMessage
}


while getopts ':ht:m:j:' option; do
  case "$option" in
    h) print_usage
       exit
       ;;
    t) tgt=$OPTARG
       ;;
    m) max_lines=$OPTARG
       ;;
    j) decrement=$OPTARG
       ;;
    :) printf "missing argument for -%s\n" "$OPTARG" >&2
       print_usage
       exit 1
       ;;
   \?) printf "illegal option: -%s\n" "$OPTARG" >&2
       print_usage
       exit 1
  esac
done
shift $((OPTIND - 1))

if [[ -z ${tgt+x}  ]]; then
  echo "Missing target language argument"
  print_usage
  exit 1
fi

if [[ -z ${max_lines+x}  ]]; then
  echo "Missing maximum lines argument"
  print_usage
  exit 1
fi

if [[ -z ${decrement+x} ]]; then
  echo "Missing decrement argument"
  print_usage
  exit 1
fi


case "$tgt" in
    hu)
    language='Hungarian'
    ;;
    de)
    language='German'
    ;;
    cz)
    language='Czech'
    ;;
    all)
    language=("hu" "cs" "de")
    ;;
esac

isarray() {
  if [[ "$(declare -p "$1")" =~ "declare -a" ]]; then
    return 0
  else
    return 1
fi
}

if isarray "language"; then
  for l in "${language[@]}"; do
      echo "Preprocessing of corpora is starting for the $l language." 
      for ((i = max_lines ; i > decrement ; i=i-decrement)); do
        $(dirname $0)/choose_corpus.sh $l $i 3000
      done
  done
else
    echo "Preprocessing of corpora is starting for the $language language." 
    for ((i = max_lines ; i > decrement ; i=i-decrement)); do
        $(dirname $0)/choose_corpus.sh $tgt $i 3000
    done
fi

./monitor_checkpoints.sh &

if isarray "language"; then
  for l in "${language[@]}"; do
      echo "Starting to train models for the $l language." 
      for ((i = max_lines ; i > decrement ; i=i-decrement)); do
        python3 training_model.py
      done
  done
else
    echo "Starting to train models for the $language language." 
    for ((i = max_lines ; i > decrement ; i=i-decrement)); do
        python3 training_model.py
    done
fi