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


sizes=('-size 75' '-size 300')
alphas=('-alpha 0.025' '-alpha 0.1')
windows=('-window 5' '-window 20')
negatives=('-negative 12' '-negative 50')
models=('-cbow 1 -sample 1e-5' '-cbow 1 -sample 1e-4' '-cbow 1 -sample 1e-3' '-cbow 0 -sample 1e-3' '-cbow 0 -sample 1e-2' '-cbow 0 -sample 1e-1')
default_parameters=('-size 150 -alpha 0.05 -window 10 -negative 25 -iter 25 -threads 1 -min_count 1 -train alldata-id.txt')
default_models=('-cbow 0 -sample 1e-2' '-cbow 1 -sample 1e-4')
mkdir time_p2v
time_fold="time_p2v/"
mkdir space_p2v
space_fold="space_p2v/"
for model in "${default_models[@]}"; do
	for size in "${sizes[@]}"; do
	    delete=("-size 150")
	    d_p=${default_parameters[@]/$delete}
	    #echo $d_p
	    #echo $size
	    d2v_out="doc2vec ""$model""$size"".txt"
	    d2v_t="$time_fold""time_""$d2v_out"
	    (time (python3 run_doc2vec_proper.py   -output "$space_fold""$d2v_out" $size $model $d_p >> "$d2v_t")) &>> "$d2v_t" &
	done
	for alpha in "${alphas[@]}"; do
	    delete=("-alpha 0.05")
	    d_p=${default_parameters[@]/$delete}
	    #echo $d_p
	    #echo $alpha
	    d2v_out="doc2vec ""$model""$alpha"".txt"
	    d2v_t="$time_fold""time_""$d2v_out"
	    (time (python3 run_doc2vec_proper.py  -output "$space_fold""$d2v_out" $alpha $model $d_p >> "$d2v_t")) &>> "$d2v_t" &
	done
	for window in "${windows[@]}"; do
	    delete=("-window 10")
	    d_p=${default_parameters[@]/$delete}
	    #echo $d_p
	    #echo $window
	    d2v_out="doc2vec ""$model""$window"".txt"
	    d2v_t="$time_fold""time_""$d2v_out"
	    (time (python3 run_doc2vec_proper.py  -output "$space_fold""$d2v_out" $window $model $d_p >> "$d2v_t")) &>> "$d2v_t" &
	done
    for negative in "${negatives[@]}"; do
	delete=("-negative 25")
	d_p=${default_parameters[@]/$delete}
	#echo $d_p
	#echo $negative
	d2v_out="doc2vec ""$model""$negative"".txt"
	d2v_t="$time_fold""time_""$d2v_out"
	(time (python3 run_doc2vec_proper.py -output "$space_fold""$d2v_out" $negative $model $d_p >> "$d2v_t")) &>> "$d2v_t" &
    done
    wait
done
for model in "${models[@]}"; do
    d_p=${default_parameters[@]}
    d2v_out="doc2vec ""$model"".txt"
    d2v_t="$time_fold""time_""$d2v_out"
    (time (python3 run_doc2vec_proper.py -output "$space_fold""$d2v_out" $model $d_p >> "$d2v_t")) &>> "$d2v_t" &
done
wait
