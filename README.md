# CDA-CS

## 1. Environment Setup

Python library dependencies:
- torch>=1.0.1
- scipy>=0.14.0
- numpy -v : 1.19.3
- gensim -v : 3.8.1
- others: sklearn

## 2. Data Preprocessing

Dataset:

Everton da S. Maldonado, Emad Shihab, Nikolaos Tsantalis: Using Natural Language Processing to Automatically Detect Self-Admitted Technical Debt. IEEE Trans. Software Eng. 43(11): 1044-1062 (2017)

## 3. Clustering

- Kmeans.py

  ```python
  python remove_words.py 
  ```

If you want to replace the dataset, you can use the variable dataset = 'dataset name' in the remove_words.py file. 

- build_graph.py

  ```python
  python build_graph.py
  ```

If you want to replace the dataset, you can use the variable dataset = 'dataset name' in the build_graph.py file. 

## 4. Data Augmentation

- eda_uad_bert_augment

  ```python
  python eda_uad_bert_augment.py --input=train.txt --output=augmented.txt --num_aug=16
  ```

You can specify your own with --output. You can also specify the number of generated augmented sentences per original sentence using --num_aug.

## 5. Test

- test.py

  ```
  python test.py
  ```

If you want to replace the dataset, you can use the variable dataset = 'dataset name' in the test.py file. 
