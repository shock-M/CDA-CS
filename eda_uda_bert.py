# coding=utf-8
# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou


import random
from random import shuffle
from tfidf import TfIdfWordRep
import eda_uda_bert_augment
from transformers import *
import tensorflow as tf
import nltk
import heapq
import re
random.seed(1)

#stop words list
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 
			'ours', 'ourselves', 'you', 'your', 'yours', 
			'yourself', 'yourselves', 'he', 'him', 'his', 
			'himself', 'she', 'her', 'hers', 'herself', 
			'it', 'its', 'itself', 'they', 'them', 'their', 
			'theirs', 'themselves', 'what', 'which', 'who', 
			'whom', 'this', 'that', 'these', 'those', 'am', 
			'is', 'are', 'was', 'were', 'be', 'been', 'being', 
			'have', 'has', 'had', 'having', 'do', 'does', 'did',
			'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
			'because', 'as', 'until', 'while', 'of', 'at', 
			'by', 'for', 'with', 'about', 'against', 'between',
			'into', 'through', 'during', 'before', 'after', 
			'above', 'below', 'to', 'from', 'up', 'down', 'in',
			'out', 'on', 'off', 'over', 'under', 'again', 
			'further', 'then', 'once', 'here', 'there', 'when', 
			'where', 'why', 'how', 'all', 'any', 'both', 'each', 
			'few', 'more', 'most', 'other', 'some', 'such', 'no', 
			'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
			'very', 's', 't', 'can', 'will', 'just', 'don', 
			'should', 'now', '', '.', '[']

#cleaning up text
import re
def get_only_chars(line):

    clean_line = ""

    line = line.replace("'", "")
    line = line.replace("'", "")
    line = line.replace("-", " ") #replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in '0123456789|._qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +',' ',clean_line) #delete extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line

########################################################################
# Synonym replacement
# Replace n words in the sentence with synonyms from wordnet
########################################################################

#for the first time you use wordnet
#import nltk
#nltk.download('wordnet')
from nltk.corpus import wordnet 

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = TFAutoModelWithLMHead.from_pretrained("bert-base-uncased", return_dict=True)

def synonym_replacement(words, n, tfidf):
	ori_words = words.copy() # 句子的复制
	new_words = []
	for word in ori_words:
		if word not in stop_words and word.isalpha():
			new_words.append(word)
	if new_words == []:
		return None
	replace_prob = tfidf.get_replace_prob(new_words)
	# print(new_words)
	# print(replace_prob)
	random_word_list = ['0', '1']
	replace_prob = replace_prob.tolist()

	#多个词
	random_word = []
	temp = replace_prob.copy()
	for i in range(3):
		index_i = temp.index(min(temp))
		temp[index_i] = float('inf')  # 将遍历过的列表最小值改为无穷大，下次不再选择
		random_word.append(new_words[index_i])

	#一个词
	#random_word = new_words[replace_prob.index(min(replace_prob))]

	# replace_index = heapq.nsmallest(3, range(len(replace_prob)))
	# for i in replace_index:
	# 	if(new_words[i].isalpha()):
	# 		random_word_list.append(new_words[i])
	# for i in range(len(new_words)):
	# 	if(new_words[i].isalpha()): #只由字母组成
	# 		if tfidf.get_random_prob() < replace_prob[i]:
	# 			# print(new_words[i])
	# 			# print(replace_prob[i])
	# 			random_word_list.append(new_words[i])
	# print(random_word_list)
	# random_word_list = list(set([word for word in words if word not in stop_words])) #句中不是停用词的词
	# random.shuffle(random_word_list)
	num_replaced = 0
	#for random_word in random_word_list: # 将选中的非停用词换掉后的句子
	if len(random_word_list) > 0:
		#random_word = random_word_list[replace_prob.index(max(replace_prob))]
		#random_word = random.choice(list(random_word_list))
		print(random_word)
		"""
		
		if num_replaced >= n:  # only replace up to n words
			break
		"""
		##################eda+uda 多个词#####################
		for temp_word in random_word:
			synonyms = get_synonyms(temp_word)  # 获取某个词所有同义词
			if len(synonyms) >= 1:
				synonym = random.choice(list(synonyms))  # 从列表中返回一个随机的同义词
				new_words = [synonym if word == temp_word else word for word in ori_words]
				# print("replaced", random_word, "with", synonym)
				num_replaced += 1

		# ##################eda+uda#####################
		# synonyms = get_synonyms(random_word)  # 获取某个词所有同义词
		# if len(synonyms) >= 1:
		# 	synonym = random.choice(list(synonyms))  # 从列表中返回一个随机的同义词
		# 	new_words = [synonym if word == random_word else word for word in ori_words]
		# 	# print("replaced", random_word, "with", synonym)
		# 	num_replaced += 1
		"""
		################eda+uda+bert#############################
		# {tokenizer.mask_token}
		new_words = ['[MASK]' if word == random_word else word for word in new_words]
		# for i in new_words:
		#	print(i)
		sequence = ' '.join(new_words)
		# sequence = 'f' + sequence
		input = tokenizer.encode(sequence, return_tensors="tf")
		mask_token_index = tf.where(input == tokenizer.mask_token_id)[0, 1]
		token_logits = model(input)[0]
		mask_token_logits = token_logits[0, mask_token_index, :]
		top_1_tokens = tf.math.top_k(mask_token_logits, 5).indices.numpy()
		for token in top_1_tokens:
			sentence = sequence.replace(tokenizer.mask_token, tokenizer.decode([token]))
			if sentence != sequence:
				break
		# print(sentence)
		new_words = sentence.split(' ')
		
		"""

	#this is stupid but we need it, trust me
	sentence = ' '.join(new_words) # 将序列中的元素以指定的字符连接生成一个新的字符串
	new_words = sentence.split(' ')

	return new_words


def _masked_language_model(model, tokenizer, sent, word_pieces, mask_id, M, N, p):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	tokenized_text = tokenizer.tokenize(sent)
	tokenized_text = ['[CLS]'] + tokenized_text
	tokenized_len = len(tokenized_text)

	tokenized_text = word_pieces + ['[SEP]'] + tokenized_text[1:] + ['[SEP]']

	if len(tokenized_text) > 512:
		tokenized_text = tokenized_text[:512]

	token_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
	segments_ids = [0] * (tokenized_len + 1) + [1] * (len(tokenized_text) - tokenized_len - 1)

	tokens_tensor = torch.tensor([token_ids]).to(device)
	segments_tensor = torch.tensor([segments_ids]).to(device)

	model.to(device)

	predictions = model(tokens_tensor, segments_tensor)

	word_candidates = torch.argsort(predictions[0, mask_id], descending=True)[:M].tolist()
	word_candidates = tokenizer.convert_ids_to_tokens(word_candidates)

	return list(filter(lambda x: x.find("##"), word_candidates))


def get_synonyms(word): # 获取所有同义词
	synonyms = set()
	for syn in wordnet.synsets(word): 
		for l in syn.lemmas(): 
			synonym = l.name().replace("_", " ").replace("-", " ").lower()
			synonym = "".join([char for char in synonym if char in ' 0123456789|.qwertyuiopasdfghjklzxcvbnm'])
			synonyms.add(synonym) 
	if word in synonyms:
		synonyms.remove(word)
	return list(synonyms)

########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################

def random_deletion(words, p, tfidf):

	#obviously, if there's only one word, don't delete it
	if len(words) == 1:
		return words

	new_words = words.copy()  # 句子的复制
	replace_prob = tfidf.get_replace_prob(new_words)
	random_word_list = []
	for i in range(len(new_words)):
		if tfidf.get_random_prob() < replace_prob[i]:
			random_word_list.append(new_words[i])
	#randomly delete words with probability p
	new_words = []
	for word in random_word_list:
		r = random.uniform(0, 1)
		if r > p:
			new_words.append(word)

	#if you end up deleting all words, just return a random word
	if len(new_words) == 0:
		rand_int = random.randint(0, len(words)-1)
		return [words[rand_int]]

	return new_words

########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################

def random_swap(words, n, tfidf):
	new_words = words.copy()
	new_words = words.copy()  # 句子的复制
	replace_prob = tfidf.get_replace_prob(new_words)
	random_word_list = []
	for i in range(len(new_words)):
		if tfidf.get_random_prob() < replace_prob[i]:
			random_word_list.append(new_words[i])

	for _ in range(n):
		new_words = swap_word(random_word_list)
	return new_words

def swap_word(new_words):
	random_idx_1 = random.randint(0, len(new_words)-1)
	random_idx_2 = random_idx_1
	counter = 0
	while random_idx_2 == random_idx_1:
		random_idx_2 = random.randint(0, len(new_words)-1)
		counter += 1
		if counter > 3:
			return new_words
	new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
	return new_words

########################################################################
# Random insertion
# Randomly insert n words into the sentence
########################################################################

def random_insertion(words, n, tfidf):
	#if (len(new_words) == 0):
	#	print("new_words: "+  new_words)
	new_words = words.copy()  # 句子的复制
	replace_prob = tfidf.get_replace_prob(new_words)
	random_word_list = []
	for i in range(len(new_words)):
		if tfidf.get_random_prob() < replace_prob[i]:
			random_word_list.append(new_words[i])
	for _ in range(n):
		add_word(new_words)
	return new_words
def add_word(new_words):
	synonyms = []
	counter = 0

	while len(synonyms) < 1:
		#if(len(new_words) == 0 ):
			#print("cnt:%d" % cnt)
		#print("new_words: %d" % len(new_words))
		random_word = new_words[random.randint(0, len(new_words)-1)]
		synonyms = get_synonyms(random_word)
		counter += 1
		if counter >= 10:
			return
	random_synonym = synonyms[0]
	random_idx = random.randint(0, len(new_words)-1)
	new_words.insert(random_idx, random_synonym)

########################################################################
# main data augmentation function
########################################################################

def eda(tfidf, sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):
	
	#sentence = get_only_chars(sentence)

	#words = sentence.split(' ')
	words = nltk.word_tokenize(sentence)
	words = [word for word in words if word is not '']
	num_words = len(words)
	
	augmented_sentences = []
	# num_new_per_technique = int(num_aug/4)+1
	num_new_per_technique = num_aug
	#sr
	if (alpha_sr > 0):
		n_sr = max(1, int(alpha_sr*num_words))
		for _ in range(num_new_per_technique):
			a_words = synonym_replacement(words, n_sr, tfidf) #words，原始句子的词
			if a_words == None:
				continue
			augmented_sentences.append(' '.join(a_words))

	#ri
	if (alpha_ri > 0):
		n_ri = max(1, int(alpha_ri*num_words))
		for _ in range(num_new_per_technique):
			a_words = random_insertion(words, n_ri, tfidf)
			augmented_sentences.append(' '.join(a_words))

	#rs
	if (alpha_rs > 0):
		n_rs = max(1, int(alpha_rs*num_words))
		for _ in range(num_new_per_technique):
			a_words = random_swap(words, n_rs, tfidf)
			augmented_sentences.append(' '.join(a_words))

	#rd
	if (p_rd > 0):
		for _ in range(num_new_per_technique):
			a_words = random_deletion(words, p_rd, tfidf)
			augmented_sentences.append(' '.join(a_words))

	#augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
	shuffle(augmented_sentences)

	#trim so that we have the desired number of augmented sentences
	if num_aug >= 1:
		augmented_sentences = augmented_sentences[:num_aug]
	else:
		keep_prob = num_aug / len(augmented_sentences)
		augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

	#append the original sentence
	# ####################eda#################
	# augmented_sentences.append(sentence)
	############ Kmeans + eda ######


	return augmented_sentences