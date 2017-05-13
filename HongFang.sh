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


default_models=('-cbow 0 -sample 1e-2' '-cbow 1 -sample 1e-4')
default_parameters=('-size 150 -alpha 0.05 -window 10 -negative 25 -iter 25 -threads 1')
min_counts=('-min_count 1' '-min_count 3' '-min_count 10')

mkdir d2v_HongJames_IMDB
d2v_IMDB_fold="d2v_HongJames_IMDB/"
mkdir C_HongJames_IMDB
C_IMDB_fold="C_HongJames_IMDB/"


for model in "${default_models[@]}"; do
    for min_count in "${min_counts[@]}"; do
        d2v_out="doc2vec ""$model""$min_count"".txt"
        python3 run_doc2vec_proper.py -output "$d2v_IMDB_fold""$d2v_out" -train alldata-id.txt $min_count $model $default_parameters &
        #python3 run_doc2vec_20ng.py -output "$_20ng_fold""$d2v_out" $min_count $model $default_parameters &    
    done
    c_out="$C_IMDB_fold""word2vec ""$model"".txt"
    delete='-threads 1'
    d_p=${default_parameters[@]/$delete}
    ./word2vec -train alldata-id.txt -output "$c_out" $model $d_p -threads 40 -binary 0 -min-count 1 -sentence-vectors 1 &
done
wait

python3 IMDB_concat_dataframe.py -classifier linearsvc -vectors "$d2v_IMDB_fold"
python3 IMDB_concat_dataframe.py -classifier lr -vectors "$d2v_IMDB_fold"

vec1="$C_IMDB_fold""word2vec ""${default_models[0]}"".txt"
vec2="$C_IMDB_fold""word2vec ""${default_models[1]}"".txt"
concat="$C_IMDB_fold""concat_word2vec_mc1"
python3 concat2models.py "$vec1" "$vec2" "$concat"
sentence_vectors="$concat""_sentence_vectors"
grep '_\*' "$concat" | sed -e 's/_\*//' | sort -n > "$sentence_vectors"

mkdir c_eval
# test
head "$sentence_vectors" -n 25000 | awk 'BEGIN{a=0;}{if (a<12500) printf "1 "; else printf "-1 "; for (b=1; b<NF; b++) printf b ":" $(b+1) " "; print ""; a++;}' > c_eval/full-train.txt
head "$sentence_vectors" -n 50000 | tail -n 25000 | awk 'BEGIN{a=0;}{if (a<12500) printf "1 "; else printf "-1 "; for (b=1; b<NF; b++) printf b ":" $(b+1) " "; print ""; a++;}' > c_eval/test.txt

iclr15/scripts/install_liblinear.sh
liblinear-1.96/train -s 0 c_eval/full-train.txt c_eval/model.logreg
liblinear-1.96/predict -b 1 c_eval/full-train.txt c_eval/model.logreg c_eval/train.logreg
liblinear-1.96/predict -b 1 c_eval/test.txt c_eval/model.logreg c_eval/out.logreg


