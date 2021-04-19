# CDA-CS

## 1. Environment Setup

Python library dependencies:
- torch>=1.0.1
- scipy>=0.14.0
- numpy
- gensim
- sklearn

## 2.Original Data
Dataset:
Wasi Uddin Ahmad, Saikat Chakraborty, Baishakhi Ray, and Kai-Wei Chang. A transformer-based approach for source code summarization.In Dan Jurafsky, Joyce Chai, and Natalie Schluter, editors, Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, ACL 2020, Online, July 5-10, 2020, pages 4998–5007. Association for Computational Linguistics, 2020.

## 3. Clustering

- tf_idf_vec.py
- 
  ```python
  python tf_idf_vec.py 
  ```

You can get the TF_IDF vetor of input file, please place this input file into the data folder, and run:

- Kmeans.py

  ```python
  python Kmeans.py 
  ```

If you want to replace the dataset, you can use the variable dataset = 'dataset name' in the build_graph.py file. 

## 4. Data Augmentation

- eda_uad_bert_augment

  ```python
  python eda_uad_bert_augment.py --input=train.txt --output=augmented.txt --num_aug=16
  ```

You can specify your own with --output. You can also specify the number of generated augmented sentences per original sentence using --num_aug.

## 5. New Dataset

- split.py

  ```
  python split.py
  ```

You need to split augmented.txt into javadoc.original and code.original_subtoken.
