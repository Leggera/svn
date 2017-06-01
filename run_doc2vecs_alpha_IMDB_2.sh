#!/bin/bash

function normalize_text {
  awk '{print tolower($0);}' < $1 | sed -e 's/\./ \. /g' -e 's/<br \/>/ /g' -e 's/"/ " /g' \
  -e 's/,/ , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' -e 's/\?/ \? /g' \
  -e 's/\;/ \; /g' -e 's/\:/ \: /g' > $1-norm
}
if [ ! -d ./data ]
then
    wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
    tar -xvf aclImdb_v1.tar.gz

    for j in train/pos train/neg test/pos test/neg train/unsup; do
	for i in `ls aclImdb/$j`; do cat aclImdb/$j/$i >> temp; awk 'BEGIN{print;}' >> temp; done
	normalize_text temp
	mv temp-norm aclImdb/$j/norm.txt
	rm temp
    done

    mkdir data
    mv aclImdb/train/pos/norm.txt data/full-train-pos.txt
    mv aclImdb/train/neg/norm.txt data/full-train-neg.txt
    mv aclImdb/test/pos/norm.txt data/test-pos.txt
    mv aclImdb/test/neg/norm.txt data/test-neg.txt
    mv aclImdb/train/unsup/norm.txt data/train-unsup.txt
fi


cat ./data/full-train-pos.txt ./data/full-train-neg.txt ./data/test-pos.txt ./data/test-neg.txt ./data/train-unsup.txt > alldata.txt
awk 'BEGIN{a=0;}{print "_*" a " " $0; a++;}' < alldata.txt > alldata-id.txt


#alphas1=('-alpha 0.01' '-alpha 0.02' '-alpha 0.03' '-alpha 0.04' '-alpha 0.05' '-alpha 0.06')
#alphas2=('-alpha 0.07' '-alpha 0.08' '-alpha 0.09' '-alpha 0.1' '-alpha 0.15' '-alpha 0.2')
default_parameters=('-size 150 -window 10 -negative 25 -threads 1 -train alldata-id.txt')
default_models1=( '-cbow 1 -sample 1e-4 -alpha 0.00005' '-cbow 1 -sample 1e-4 -alpha 0.0001' 
'-cbow 1 -sample 1e-4 -alpha 0.0005' '-cbow 1 -sample 1e-4 -alpha 0.001')
iters=('-iter 25')
min_count=('-min_count 1')

mkdir time_d2v
time_fold="time_d2v/"
mkdir space_d2v
space_fold="space_d2v/"
for iter in "${iters[@]}";do
  for m_c in "${min_count[@]}"; do
      for model in "${default_models1[@]}"; do
	d2v_out="doc2vec ""$model"" $iter"" $m_c"" .txt"
	d2v_t="$time_fold""time_""$d2v_out"
	(time (python3 run_doc2vec_proper.py  -output "$space_fold""$d2v_out" $iter $m_c $model $default_parameters >> "$d2v_t")) &>> "$d2v_t" &
      done
  done
done
wait
