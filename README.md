# CDA-CS

## 1. Environment Setup

Python library dependencies:
- torch>=1.0.1
- scipy>=0.14.0
- numpy
- gensim
- NLTK
- sklearn

## 2. Original Data
Dataset:
Wasi Uddin Ahmad, Saikat Chakraborty, Baishakhi Ray, and Kai-Wei Chang. A transformer-based approach for source code summarization. ACL 2020, Online, July 5-10, 2020, pages 4998–5007. Association for Computational Linguistics, 2020.

## 3. Clustering

- tf_idf_vec.py

  ```python
  python tf_idf_vec.py 
  ```

You can get the TF_IDF vetor of input file, please place this input file into the data folder, and run:

- K-means.py

  ```python
  python K-means.py 
  ```

In this way, the data that needs to be data augmentation can be obtained.

## 4. Data Augmentation

- eda_uad_bert_augment

  ```python
  python eda_uad_bert_augment.py --input=train.txt --output=augmented.txt --num_aug=20
  ```

You can specify your own with --output. You can also specify the number of generated augmented sentences per original sentence using --num_aug.

## 5. New Dataset

- split.py

  ```
  python split.py
  ```

You need to split augmented.txt into javadoc.original and code.original_subtoken.
