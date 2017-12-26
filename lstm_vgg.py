#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os

from vqa_api.PythonHelperTools.vqaTools.vqa import VQA
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import image
from tqdm import tqdm

from tools import ann_file, ques_file, img_dir, img_file

class LSTMVGG():

    def __init__(self, dataDir, versionType="", taskType="OpenEnded", dataType="mscoco",
                 dataSubTypesTrain=["train2014"], dataSubTypeTest="val2014", 
                 n_train=None, n_test=None):

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
        # Number of questions for each training set
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

        # Merge the annotations of the two different dataSubTypes
        self.annotations_train = getattr(self, "vqa_{}".format(dataSubTypes[0])).dataset["annotations"]
        if len(dataSubTypes) > 1:
            print "--> Merging the annotations of the different dataSubTypes"
            for dataSubType in dataSubTypes[1:]:
                self.annotations_train += getattr(self, "vqa_{}".format(dataSubType)).dataset["annotations"]

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
            dic["data_subtype"] = dataSubTypeTest

        # Test annotations
        self.annotations_test = self.vqa_test.dataset["annotations"]

        # Reduce the train and test set 
        if n_train:
            self.questions_train = self.questions_train[:n_train]
        if n_test:
            self.questions_test = self.questions_train[:n_test]

    def tokenize_questions(self):
        """ Train a tokenizer on the training set questions.
        """
        # List of questions
        self.train_questions = [dic["question"]
                     for dic in self.questions_train]
        self.test_questions = [dic["question"]
                     for dic in self.questions_test]

        # Create the tokenizer
        tokenizer = Tokenizer()

        # Fit the tokenizer on the training set questions
        tokenizer.fit_on_texts(train_questions)

        # Vocabulary size of the training set
        self.vocabulary_size_train = len(tokenizer.word_index.keys())

        # Let's use the embedding that has been fit on the training data
        train_sequences = tokenizer.texts_to_sequences(train_questions)
        test_sequences = tokenizer.texts_to_sequences(test_questions)
        
        # Longest sentence in the training set
        self.input_length = max([len(sequence) for sequence in train_sequences])

        # Pad the sequences to a maximum length
        self.train_sequences = pad_sequences(train_sequences, 
                                               maxlen=self.input_length, 
                                               padding="post")
        self.test_sequences = pad_sequences(test_sequences, 
                                               maxlen=self.input_length, 
                                               padding="post")
    def process_images(self):
        train_image_ids = list(set((dic["image_id"], dic["data_subtype"]) 
                           for dic in self.questions_train))
        test_image_ids = list(set((dic["image_id"], dic["data_subtype"]) 
                           for dic in self.questions_test))

        self.train_images = self.process_images_(train_image_ids)
        self.test_images = self.process_images_(test_image_ids)

    def process_images_(self, image_ids):
        """Resize images accordingly to VGG16 input, and preprocess them.
        
        Parameters
        ----------
        image_ids : list(tupe)
            Each tuple is (image_id, data_subtype). The image ids are unique.
        """
        # List to store the arrays
        imgs = []

        for image_id in tqdm(image_ids):
            img = image.load_img(os.path.join(img_dir(self.dataDir, self.dataType, 
                image_id[1]), img_file(image_id[1], image_id[0])), target_size=(224, 224)) 
            img = image.img_to_array(img)
            imgs.append(img)
        imgs = np.stack(imgs)
        imgs = preprocess_input(imgs)   

        # Duplicate the array for each image three times to correspond to the 
        # number of training questions
        imgs = np.repeat(imgs, 3, axis=0)

        return imgs

    def get_top_answers(self, top_n=1000):
        self.answers = [annotation["answers"][i]["answer"] for i in range(10) 
        for annotation in self.annotations]
        self.top_answers = [answer for (answer, occ) in Counter(self.answers).most_common(1000)]




