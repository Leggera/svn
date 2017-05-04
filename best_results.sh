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


mkdir best_IMDB
IMDB_fold="best_IMDB/"
mkdir best_20ng
_20ng_fold="best_20ng/"
default_models=('-cbow 0 -sample 1e-2' '-cbow 1 -sample 1e-4')
default_parameters=('-size 150 -alpha 0.05 -window 10 -negative 25 -iter 25 -threads 1 -min_count 5 -train alldata-id.txt')
for model in "${default_models[@]}"; do
    delete=("-alpha 0.05")
    d_p=${default_parameters[@]/$delete}
    alpha = "-alpha 0.025"
    d_p=("${n_p[@]}" $alpha)
    python3 run_doc2vec_proper.py -output "$IMDB_fold""$d2v_out" $alpha $model $d_p &
    delete=("-window 10")
    n_p=${default_parameters[@]/$delete}
    window = "-window 5"
    n_p=("${n_p[@]}" $window)
    python3 run_doc2vec_20ng.py -output "$_20ng_fold""$d2v_out" $window $model $n_p &
done
python3 BagOfNgrams_Iclmdb_words_and_bigrams.py
python3 BagOfNgrams_20ng_words_and_bigrams.py

python3 IMDB_concat_dataframe.py -classifier lr -vectors IMDB_fold
python3 IMDB_concat_dataframe.py -classifier linearsvc -vectors IMDB_fold
python3 20ng_concat_dataframe.py -classifier lr -vectors _20ng_fold
python3 20ng_concat_dataframe.py -classifier linearsvc -vectors _20ng_fold
