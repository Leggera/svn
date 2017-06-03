#!/bin/bash
git clone https://github.com/mesnilgr/iclr15.git

cp iclr15/scripts/word2vec.c .
gcc word2vec.c -o word2vec -lm -pthread -O3 -march=native -funroll-loops

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


sizes=('-size 75')
alphas=('-alpha 0.025' '-alpha 0.1')
windows=('-window 5' '-window 20')
negatives=('-negative 12' '-negative 50')
iters=('-iter 25')

models=('-cbow 1 -sample 1e-5' '-cbow 1 -sample 1e-4' '-cbow 1 -sample 1e-3' '-cbow 1 -sample 1e-2'
	'-cbow 0 -sample 1e-4' '-cbow 0 -sample 1e-3' '-cbow 0 -sample 1e-2' '-cbow 0 -sample 1e-1')
#sizes=('-size 1' '-size 3')
#alphas=('-alpha 0.025' '-alpha 0.1')
#windows=('-window 1' '-window 2')
#negatives=('-negative 1' '-negative 2')
#iters=('-iter 2')
default_parameters=('-size 150 -alpha 0.05 -window 10 -negative 25 -threads 1 -train alldata-id.txt')
default_models=('-cbow 0 -sample 1e-2' '-cbow 1 -sample 1e-4')
min_count=('-min-count 1')

mkdir time_w2v
time_fold="time_w2v/"
mkdir space_w2v
space_fold="space_w2v/"
for iter in "${iters[@]}";do
for m_c in "${min_count[@]}"; do
    for model in "${default_models[@]}"; do
	for size in "${sizes[@]}"; do
	    delete=("-size 150")
	    d_p=${default_parameters[@]/$delete}
	    #echo $d_p
	    #echo $size
	    d2v_out="word2vec ""$model"" $size"" $iter"" $m_c"" .txt"
	    d2v_t="$time_fold""time_""$d2v_out"
	    (time (./word2vec -output "$space_fold""$d2v_out" $iter $m_c $size $model $d_p -binary 0 -sentence-vectors 1 >> "$d2v_t")) &>> "$d2v_t" &	    
	done
	for alpha in "${alphas[@]}"; do
	    delete=("-alpha 0.05")
	    d_p=${default_parameters[@]/$delete}
	    #echo $d_p
	    #echo $alpha
	    d2v_out="word2vec ""$model"" $alpha"" $iter"" $m_c"" .txt"
	    d2v_t="$time_fold""time_""$d2v_out"
	    (time (./word2vec -output "$space_fold""$d2v_out" $iter $m_c $alpha $model $d_p -binary 0 -sentence-vectors 1 >> "$d2v_t")) &>> "$d2v_t" &
	done
	for window in "${windows[@]}"; do
	    delete=("-window 10")
	    d_p=${default_parameters[@]/$delete}
	    #echo $d_p
	    #echo $window
	    d2v_out="word2vec ""$model"" $window"" $iter"" .txt"
	    d2v_t="$time_fold""time_""$d2v_out"
	    (time (./word2vec -output "$space_fold""$d2v_out" $iter $m_c $window $model $d_p -binary 0 -sentence-vectors 1 >> "$d2v_t")) &>> "$d2v_t" &
	done
	for negative in "${negatives[@]}"; do
	    delete=("-negative 25")
	    d_p=${default_parameters[@]/$delete}
	    #echo $d_p
	    #echo $negative
	    d2v_out="word2vec ""$model"" $negative"" $iter"" $m_c"" .txt"
	    d2v_t="$time_fold""time_""$d2v_out"
	    (time (./word2vec -output "$space_fold""$d2v_out" $iter $m_c $negative $model $d_p -binary 0 -sentence-vectors 1 >> "$d2v_t")) &>> "$d2v_t" &
	done
    done
    for model in "${models[@]}"; do
	d_p=${default_parameters[@]}
	d2v_out="word2vec ""$model"" $m_c"" $iter"" .txt"
	d2v_t="$time_fold""time_""$d2v_out"
	(time (./word2vec -output "$space_fold""$d2v_out" $model $iter $m_c $d_p -binary 0 -sentence-vectors 1 >> "$d2v_t")) &>> "$d2v_t" &
    done
done
done
wait
