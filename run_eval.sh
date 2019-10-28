judgement=$1
output=$2

./trec_eval.8.1/trec_eval -q ${judgement} ${output} > ${output}.treceval
tail -29 ${output}.treceval | grep -e '^map' -e 'recip_rank' -e 'P30'
exit 0