run_doc2vecs_20ng.sh обучает doc2vec на 20newsgroups (вызывая run_doc2vec_20ng.py)
run_doc2vecs_IMDB.sh обучает doc2vec на IMDB (вызывая run_doc2vec_proper.py)

BagOfNgrams_20ng.py строит вектора модели BagOfNgrams с N = 1 на 20newsgroups и подбирает оптимальные параметры с помощью GridSearch
BagOfNgrams_20ng_words_and_bigrams.py строит вектора модели BagOfNgrams с N = (1, 2) на 20newsgroups и подбирает оптимальные параметры с помощью GridSearch
BagOfNgrams_Iclmdb.py строит вектора модели BagOfNgrams с N = 1 на IMDB и подбирает оптимальные параметры с помощью GridSearch
BagOfNgrams_Iclmdb_words_and_bigrams.py строит вектора модели BagOfNgrams с N = (1, 2) на IMDB и подбирает оптимальные параметры с помощью GridSearch

Скрипты BagOfNgrams*.py очень похожи между собой, поэтому комментарии написаны только к BagOfNgrams_20ng.py


IMDB_concat_dataframe.py подбирает оптимальные параметры для векторов doc2vec IMDB с помощью GridSearch для каждой комбинации конкатенаций (17 штук)
20ng_concat_dataframe.py подбирает оптимальные параметры для векторов doc2vec 20newsgroups с помощью GridSearch для каждой комбинации конкатенаций (17 штук)

Для этих двух скриптов в качестве вхожных параметров должен быть указан путь к векторам (параметр vectors)б классификатор (classifier: lr или linearsvc) и опционально величина регуляризации С (иначе она подбирается из (10**-5, 3*10**-5, 10**-4, 3*10**-4)).

Все подобранные параметры классификаторов и достигнутые с помощью них accuracy, precision, recall и f1-score (для train и test), а также точность на cross validation записываются в таблицу напротив соответствующих значений параметров paragraph2vec (для каждого классификатора отдельно). Каждая таблица сохраняется в своей директории, в названии которой указана дата начала классификации и параметры рассматриваемых векторов.

Plotting Dataframe.ipynb сохраняет все графики для всех датасетов, для всех мер в отдельные директории (только для doc2vec)
Например, в concat_20ng/precision/ будут хранится графики изменения точности для всех параметров doc2vec (alpha, negative, etc.) на выборке 20newsgroups (при конкатенации PV-DBOW и PV-DM)
