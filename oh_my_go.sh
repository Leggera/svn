chmod +x *.sh
./run_doc2vecs_IMDB.sh
./run_doc2vecs_20ng.sh
python3 IMDB_concat_dataframe.py -classifier lr
python3 IMDB_concat_dataframe.py -classifier linearsvc
python3 20ng_concat_dataframe.py -classifier lr
python3 20ng_concat_dataframe.py -classifier linearsvc
