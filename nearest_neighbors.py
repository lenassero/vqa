#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cPickle as pkl
import numpy as np
import sys
import os
from tqdm import tqdm

from tools import ann_file, ques_file, img_dir, skipthoughts_npy_file, skipthoughts_idx_to_qid_file, skipthoughts_test_qid_to_train_knn_qids_file,save_results
from vqa_api.PythonHelperTools.vqaTools.vqa import VQA
from skip_thoughts import skipthoughts

from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from tools import img_dir, img_file, ann_file, ques_file

import keras
from keras.layers import Dense, Dropout, LSTM, multiply
from keras.layers.embeddings import Embedding
from keras.utils import plot_model
from keras.layers import Input
from keras.models import Model
from collections import Counter




class NearestNeighbors():

	def __init__(self, dataDir, versionType="", taskType="OpenEnded", dataType="mscoco",
				 dataSubTypesTrain=["train2014"], dataSubTypeTest="val2014", 
				 n_train=None):

		self.dataDir = dataDir
		self.versionType = versionType
		self.taskType = taskType
		self.dataType = dataType

		if type(dataSubTypesTrain) == list:
			self.dataSubTypesTrain = dataSubTypesTrain
		else:
			self.dataSubTypesTrain = [dataSubTypesTrain]

		self.dataSubTypeTest = dataSubTypeTest

		# File names for the annotations and the questions
		for dataSubType in dataSubTypesTrain:
			setattr(self, "annFile_{}".format(dataSubType), 
					ann_file(dataDir, versionType, dataType, dataSubType))
			setattr(self, "quesFile_{}".format(dataSubType), 
					ques_file(dataDir, versionType, taskType, dataType, dataSubType))
			print(self.annFile_train2014)

		# Initialize VQA api for each dataSubType
		for dataSubType in dataSubTypesTrain:
			print "--> {}".format(dataSubType)
			setattr(self, "vqa_{}".format(dataSubType),
					VQA(getattr(self, "annFile_{}".format(dataSubType)), 
						getattr(self, "quesFile_{}".format(dataSubType))))

		# Merge the questions of the two different dataSubTypesTrain
		self.questions_train = getattr(
			self, "vqa_{}".format(dataSubTypesTrain[0])).questions["questions"]
		for dic in self.questions_train:
			dic["data_subtype"] = dataSubTypesTrain[0]

		if len(dataSubTypesTrain) > 1:
			print "--> Merging the annotations of the different dataSubTypesTrain"
			for dataSubType in dataSubTypesTrain[1:]:
				questions = getattr(self, "vqa_{}".format(dataSubType))\
									.questions["questions"]
				for dic in questions:
					dic["data_subtype"] = dataSubType
				self.questions_train += questions

		# Reduce the size of the train set
		if n_train:
			self.questions_train = self.questions_train[:n_train]
		self.n_train = n_train

		# Load the skipthoughts model and initialize the encoder
		self.model = skipthoughts.load_model()
		self.encoder = skipthoughts.Encoder(self.model)

		# Load vggnet fc7 layer
		vgg_model = keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', 
							   input_tensor=None, input_shape=None, 
							   pooling=None, classes=1000)
		self.fc7 = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer("fc2").output)

	def encode_train_questions(self):
		"""Encode questions from the training set.
		"""

		# File name to save the array of vectors
		vectors_train_npy_file = skipthoughts_npy_file(
			self.dataDir, self.taskType, self.dataType, self.dataSubTypesTrain, self.n_train)
		vectors_train_idx_to_qid_file = skipthoughts_idx_to_qid_file(
			self.dataDir, self.taskType, self.dataType, self.dataSubTypesTrain, self.n_train)

		if os.path.exists(vectors_train_npy_file) and os.path.exists(vectors_train_idx_to_qid_file):
			self.vectors_train = np.load(vectors_train_npy_file)
			with open(vectors_train_idx_to_qid_file, "r") as f:
				self.vectors_train_idx_to_qid = pkl.load(f)

		else:
			# List of questions and their corresponding ids
			questions = [dic["question"]
						 for dic in self.questions_train]
			question_ids = [dic["question_id"]
							for dic in self.questions_train]

			# Encode the questions of the train set
			self.vectors_train = self.encoder.encode(questions)

			# Number of questions encoded
			n_questions_train = len(self.vectors_train)

			# Dictionary mapping the indices in vectors to the question ids
			self.vectors_train_idx_to_qid = {i: question_id for (
				i, question_id) in zip(range(n_questions_train), question_ids)}

			# Save the encoded vectors
			np.save(vectors_train_npy_file, self.vectors_train)

			# Save the mapping
			with open(vectors_train_idx_to_qid_file, "w") as f:
				pkl.dump(self.vectors_train_idx_to_qid, f)

	def encode_test_questions(self, n_test=None):
		"""Encode questions from the test set.
		"""
		# File name to save the array of vectors
		vectors_test_npy_file = skipthoughts_npy_file(
			self.dataDir, self.taskType, self.dataType, self.dataSubTypeTest, n_test)
		vectors_test_idx_to_qid_file = skipthoughts_idx_to_qid_file(
			self.dataDir, self.taskType, self.dataType, self.dataSubTypeTest, n_test)


		if os.path.exists(vectors_test_npy_file) and os.path.exists(vectors_test_idx_to_qid_file):
			self.vectors_test = np.load(vectors_test_npy_file)
			with open(vectors_test_idx_to_qid_file, "r") as f:
				self.vectors_test_idx_to_qid = pkl.load(f)

		else:

			# File names for the annotations and questions
			self.annFile_test = ann_file(
				self.dataDir, self.versionType, self.dataType, self.dataSubTypeTest)
			self.quesFile_test = ques_file(
				self.dataDir, self.versionType, self.taskType, self.dataType, self.dataSubTypeTest)

			# Initialize VQA api for the self.dataSubTypeTest on which to make predictions
			# on the answers
			self.vqa_test = VQA(self.annFile_test, self.quesFile_test)
			self.questions_test = self.vqa_test.questions["questions"]
			for dic in self.questions_test:
				dic["data_subtype"] = self.dataSubTypeTest

			# Reduce the size of the test set
			self.questions_test = self.questions_test[:n_test]
			self.n_test = n_test

			# List of questions and their corresponding ids
			questions = [dic["question"]
						 for dic in self.questions_test]
			question_ids = [dic["question_id"]
							for dic in self.questions_test]

			# Encode the questions of the train set
			self.vectors_test = self.encoder.encode(questions)

			# Number of questions encoded
			n_questions_test = len(self.vectors_test)

			# Dictionary mapping the indices in vectors to the question ids
			# and vice-versa
			self.vectors_test_idx_to_qid = {i: question_id for (
				i, question_id) in zip(range(n_questions_test), question_ids)}
			self.vectors_test_qid_to_idx = {
				v: k for (k, v) in self.vectors_test_idx_to_qid.iteritems()}

			# Save the encoded vectors
			np.save(vectors_test_npy_file, self.vectors_test)

			# Save the mapping
			with open(vectors_test_idx_to_qid_file, "w") as f:
				pkl.dump(self.vectors_test_idx_to_qid, f)

	def compute_cosine_similarity(self, test_vectors, train_vectors):

		test_norms = np.linalg.norm(test_vectors, axis=1)
		train_norms = np.linalg.norm(train_vectors, axis=1)

		cosine_similarities = test_vectors.dot(
			train_vectors.T)/(np.multiply(test_norms, train_norms))

		return cosine_similarities

	def nearest_neighbors(self, k=4):

		test_qid_to_nn_train_qids_file = skipthoughts_test_qid_to_train_knn_qids_file(
			self.dataDir, self.taskType, self.dataType, self.dataSubTypesTrain, 
			self.dataSubTypeTest, k, self.n_train, self.n_test)

		if os.path.exists(test_qid_to_nn_train_qids_file):
			with open(test_qid_to_nn_train_qids_file, "r") as f:
				test_qid_to_nn_train_qids = pkl.load(f)

		else:

			# Compute the cosine similarity between each test vector and each train
			# vector
			cs = self.compute_cosine_similarity(
				self.vectors_test, self.vectors_train)

			# Sort the scores in ascending order and return the indices
			cs_sorted_args = np.flip(np.argsort(cs, axis=1), 1)[:, :k]

			# Convert the train array indices to the question ids
			train_idx_to_qid = np.vectorize(
				lambda idx: self.vectors_train_idx_to_qid[idx])
			cs_sorted_args = train_idx_to_qid(cs_sorted_args)

			# Dictionnary mapping the each question id in the test set to the k
			# most similar questions (question ids) in the training set. The latters
			# are stored in an array
			test_qid_to_nn_train_qids = {self.vectors_test_idx_to_qid[idx]:
										 cs_sorted_args[idx]
										 for idx in range(cs_sorted_args.shape[0])}

			with open(test_qid_to_nn_train_qids_file, "w") as f:
				pkl.dump(test_qid_to_nn_train_qids, f)

		return test_qid_to_nn_train_qids

	def encode_train_images(self):
		# Encode the training images
		image_list_train = [(dic["image_id"], dic["data_subtype"]) for dic in self.questions_train]
		imgs_train = []
		dataType = "mscoco"
		train_sliced = image_list_train[::3]
		for image_id in tqdm(train_sliced):
			# Resize the images as VGG inputs
			img_train = image.load_img(os.path.join(img_dir(self.dataDir, dataType, 
				image_id[1]), img_file(image_id[1], image_id[0])), target_size=(224, 224))
			img_train = image.img_to_array(img_train)
			imgs_train.append(img_train)

		imgs_train = np.stack(imgs_train)
		# Preprocess the images corresponding to VGG
		imgs_train = preprocess_input(imgs_train)
		# Encoding
		encoded_img_train = self.fc7.predict(imgs_train)
		encoded_img_train = np.repeat(encoded_img_train, 3, axis = 0)
		# Dictionary mapping image_id to the fc7 encoding of the image in the training
		self.fc7_train = {image_list_train[i][0]:encoded_img_train[i,] for i in range(np.size(encoded_img_train,0))}


	def encode_test_images(self):
		# Encode the test images
		image_list_test = [(dic["image_id"], dic["data_subtype"]) for dic in self.questions_test]
		imgs_test = []
		dataType = "mscoco"
		test_sliced = image_list_test[::3]
		for image_id in tqdm(test_sliced):
			# Resize the images as VGG inputs
			img_test = image.load_img(os.path.join(img_dir(self.dataDir, dataType, 
				image_id[1]), img_file(image_id[1], image_id[0])), target_size=(224, 224))
			img_test = image.img_to_array(img_test)
			imgs_test.append(img_test)
		imgs_test = np.stack(imgs_test)
		# Preprocess the images corresponding to VGG
		imgs_test = preprocess_input(imgs_test)
		# Encoding
		encoded_img_test = self.fc7.predict(imgs_test)
		encoded_img_test = np.repeat(encoded_img_test, 3, axis = 0)
		# Dictionary mapping image_id to the fc7 encoding of the image in the test
		self.fc7_test = {image_list_test[i][0]:encoded_img_test[i,] for i in range(np.size(encoded_img_test,0))}




	def top_images(self):

		dic_nn = self.nearest_neighbors(k=4)

		# Dictionaries mapping question_id to image_id in train & test
		train_qID_to_imgID={dic["question_id"]:dic["image_id"] for dic in self.questions_train}
		test_qID_to_imgID = {dic["question_id"]:dic["image_id"] for dic in self.questions_test}

		# Dictionary mapping (question_id, image_id) in test to the array of image_id
		# in train
		dic_imgID_nn = {(qID, test_qID_to_imgID[qID]):np.array([train_qID_to_imgID[q] for q in dic_nn[qID]])
		for qID in dic_nn.keys()}

		# Dictionary mapping question_id from test to (fc7 from test, array of fc7 features from train)
		fc7_dic = {q[0]:(self.fc7_test[q[1]], np.array([self.fc7_train[e] for e in dic_imgID_nn[q]]))
		for q in dic_imgID_nn.keys()}

		# Dictionary of cosine similarities
		cs_dic = {qID: np.divide(fc7_dic[qID][0].dot(fc7_dic[qID][1].T),np.linalg.norm(fc7_dic[qID][0])*np.linalg.norm(fc7_dic[qID][1], axis=1))
		for qID in fc7_dic.keys()}


		# Dictionary mapping question_id from test to index in the array of knn questions
		qID_to_img_nn_index = {qID: np.argmax(cs_dic[qID])
		for qID in cs_dic.keys()}

		# Dictionary mapping question_id from test to top_image in train
		qID_to_img_nn = {qID: dic_imgID_nn[(qID,test_qID_to_imgID[qID])][np.argmax(cs_dic[qID])]
		for qID in cs_dic.keys()}

		# Dictionary mapping question_id from test to the sorted list of NN img in train
		sorted_qID_to_img_nn = {qID: dic_imgID_nn[(qID,test_qID_to_imgID[qID])][np.argsort(cs_dic[qID])]
		for qID in cs_dic.keys()}


		return qID_to_img_nn_index, qID_to_img_nn, sorted_qID_to_img_nn


	def predict(self, dataSubType):

	    # File names for the annotations and questions
	    self.annFile_test = ann_file(self.dataDir, self.versionType, self.dataType, dataSubType)
	    self.quesFile_test = ques_file(self.dataDir, self.versionType, self.taskType, self.dataType, dataSubType)

	    # Initialize VQA api for the dataSubType on which to make predictions 
	    # on the answers
	    self.vqa_test = VQA(self.annFile_test, self.quesFile_test)

	    dic_nn = self.nearest_neighbors(k=4) # q_id in train to array of q_ids in train
	    qID_to_img_nn_index, _, _ = self.top_images()

	    train_qID_to_imgID={dic["question_id"]:dic["image_id"] for dic in self.questions_train}

	    # Dicionary mapping q_id in train to its answer
	    answers_train = {annotation["question_id"]:Counter((annotation["answers"][i]["answer"]
	    	for i in range(10))).most_common(1)[0][0] 
	    for annotation in self.vqa_train2014.dataset["annotations"]}


	    # Result list [{"answer": "picked from NN image", "question_id": 1}] 
	    res = [{"answer": answers_train[dic_nn[q_id][qID_to_img_nn_index[q_id]]] , "question_id": q_id} 
	    for q_id in qID_to_img_nn_index.keys()]
	    # Save the results
	    save_results(res, self.dataDir, self.taskType, self.dataType, dataSubType, 
	                 self.__class__.__name__)
        










