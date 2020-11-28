#preprocessing test files from the news corpus 2018 has test files for all languages except hungarian
#test files then come from the 2009 data
function die {
	echo "$1" >&2
	exit 1
}
if [[ $# -ne 2 ]]; then
	die "USAGE: $0 src tgt"
fi
src=$1
tgt=$2
max_lines=2500000
orig=orig_$max_lines/$tgt
max_test_lines=3000
tmp=tmp
if [[ ! -d "$tmp" ]]; then
	mkdir "$tmp"
fi
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
    if [[ $max_test_lines -lt 0 ]]; then
        filtered=$num_lines
        ln -sf $(realpath "$testfile") "$testfile.s.short"
    else
        filtered=$max_test_lines
        head -n "$filtered" "$testfile" > "$testfile.s.short"
    fi
    #preprocesses sgm files to strip them from xml tags
    grep '<seg id' $testfile.s.short | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g"  | \
				sed -e 's/$/\n/g' > $orig/test_deseg.$l
        # tokenize $tmp/test_temp.$l $l $tmp/test.$l
    echo ""
done