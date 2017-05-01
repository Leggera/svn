#!/bin/bash

function normalize_text {
  awk '{print tolower($0);}' < $1 | sed -e 's/\./ \. /g' -e 's/<br \/>/ /g' -e 's/"/ " /g' \
  -e 's/,/ , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' -e 's/\?/ \? /g' \
  -e 's/\;/ \; /g' -e 's/\:/ \: /g' > $1-norm
}

sizes=('-size 75' '-size 300')
alphas=('-alpha 0.025' '-alpha 0.1')
windows=('-window 5' '-window 20')
negatives=('-negative 12' '-negative 50')
models=('-cbow 1 -sample 1e-5' '-cbow 1 -sample 1e-4' '-cbow 1 -sample 1e-3' '-cbow 0 -sample 1e-3' '-cbow 0 -sample 1e-2' '-cbow 0 -sample 1e-1')
default_parameters=('-size 150 -alpha 0.05 -window 10 -negative 25 -iter 25 -threads 1 -min_count 5')
default_models=('-cbow 0 -sample 1e-2' '-cbow 1 -sample 1e-4')
mkdir time_p2v_20ng_combine
time_fold="time_p2v_20ng_combine/"
mkdir space_p2v_20ng_combine
space_fold="space_p2v_20ng_combine/"

for model in "${default_models[@]}"; do
  for negative in "${negatives[@]}"; do
    delete=("-negative 25")
    d_p=${default_parameters[@]/$delete}
    d_p=("${d_p[@]}" "$negative")
    echo "1"
    echo $d_p
    for window in "${windows[@]}"; do
      delete=("-window 10")
      n_p=${d_p[@]/$delete}
      n_p=("${n_p[@]}" "$window")
      d2v_out="doc2vec ""$model""$n_p"".txt"
      d2v_t="$time_fold""time_""$d2v_out"
      echo "2"
      echo $n_p
      (time (python3 run_doc2vec_20ng.py -output "$space_fold""$d2v_out" $model $n_p >> "$d2v_t")) &>> "$d2v_t" &
    done
    for alpha in "${alphas[@]}"; do
        delete=("-alpha 0.05")
        n_p=${d_p[@]/$delete}
        n_p=("${n_p[@]}" "$alpha")
        d2v_out="doc2vec ""$model""$n_p"".txt"
        d2v_t="$time_fold""time_""$d2v_out"
        echo "3"
        echo $n_p
        (time (python3 run_doc2vec_20ng.py -output "$space_fold""$d2v_out" $model $n_p >> "$d2v_t")) &>> "$d2v_t" &
    done
    for size in "${sizes[@]}"; do
        delete=("-size 150")
        n_p=${d_p[@]/$delete}
        n_p=("${n_p[@]}" "$size")
        d2v_out="doc2vec ""$model""$n_p"".txt"
        d2v_t="$time_fold""time_""$d2v_out"
        echo "4"
        echo $n_p
        (time (python3 run_doc2vec_20ng.py -output "$space_fold""$d2v_out" $model $n_p >> "$d2v_t")) &>> "$d2v_t" &
    done
  done
done
