DATASET=$1 # "TrecQA", 'twitter', "Quora", "TwitterURL"
join=$2 # "matching"(MPHCNN), "biattention", "hcan"
device=$3
TEST="none"
if [ "${DATASET}" = "twitter" ] ; then
  TEST=$4
fi

if [ ! -d "tune_logs" ] ; then
  mkdir tune_logs
fi
# Prepare Dataset if data is not created yet
# python -u train.py -dataset $DATASET -j $join --epochs 0 &> tune-logs/deep_${DATASET}.log;

encoder='deepconv'
optimizer='sgd'
lr=0.05
embed='word2vec'
batch_size=256
# Parameter Search
for weight in 'none' 'query'
do
  for model in "word_only" "complete"
  do
    for nb_layers in 4 5 6
    do
      for nb_filters in 256 128 512
      do
        for dropout in 0.1 0.2 0.3 0.4
        do
          CUDA_VISIBLE_DEVICES=$device python -u train.py -e ${encoder} --nb_layers ${nb_layers} --model_option $model --dataset $DATASET -j $join -w $weight -l -b ${batch_size} -n ${nb_filters} -d ${dropout} -o ${optimizer} --lr ${lr} --epochs 20 -t $TEST -v 2 --emb ${embed} &> tune_logs/deep_${DATASET}_trainall_${TEST}_m${model}_nbfilter${nb_filters}_nblayer${nb_layers}_d${dropout}_ttrue_b${batch_size}_o${optimizer}_lr${lr}_w${weight}_mfalse_j${join}_e${embed}.log ;
        done
      done
    done
  done
done
