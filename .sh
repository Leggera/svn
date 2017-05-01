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
mkdir time_p2v_20ng
time_fold="time_p2v_20ng/"
mkdir space_p2v_20ng
space_fold="space_p2v_20ng/"

for model in "${default_models[@]}"; do
  for negative in "${negatives[@]}"; do
    delete=("-negative 25")
    d_p=${default_parameters[@]/$delete}
    d_p=("${d_p[@]}" negative)
    for window in "${windows[@]}"; do
      delete=("-window 10")
      d_p=${d_p[@]/$delete}
      d_p=("${d_p[@]}" window)
      for alpha in "${alphas[@]}"; do
        delete=("-alpha 0.05")
        d_p=${d_p[@]/$delete}
        d_p=("${d_p[@]}" alpha)
        for size in "${sizes[@]}"; do
            delete=("-size 150")
            d_p=${d_p[@]/$delete}
            d_p=("${d_p[@]}" size)
            d2v_out="doc2vec ""$model""$d_p"".txt"
            d2v_t="$time_fold""time_""$d2v_out"
            (time (python3 run_doc2vec_20ng.py -output "$space_fold""$d2v_out" $model $d_p >> "$d2v_t")) &>> "$d2v_t" &
        done
      done
    done
  done
done
