#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os

from vqa_api.PythonHelperTools.vqaTools.vqa import VQA
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing import image
from collections import Counter
from tqdm import tqdm

from tools import ann_file, ques_file, img_dir, img_file

class LSTMVGG():

    def __init__(self, dataDir, versionType="", taskType="OpenEnded", dataType="mscoco",
                 dataSubTypesTrain=["train2014"], dataSubTypeTest="val2014"):
        """Load train and test questions and annotations using the VQA API.
        
        Parameters
        ----------
        dataDir : TYPE
            Description
        versionType : str, optional
            Description
        taskType : str, optional
            Description
        dataType : str, optional
            Description
        dataSubTypesTrain : list, optional
            Description
        dataSubTypeTest : str, optional
            Description
        """
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
        self.annotations_train = getattr(self, "vqa_{}".format(dataSubTypesTrain[0])).dataset["annotations"]
        if len(dataSubTypesTrain) > 1:
            print "--> Merging the annotations of the different dataSubTypesTrain"
            for dataSubType in dataSubTypesTrain[1:]:
                self.annotations_train += getattr(self, "vqa_{}".format(dataSubTypeTrain)).dataset["annotations"]

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

    def tokenize_questions(self):
        """ Train a tokenizer on the training set questions and tokenize the 
        train and set questions. The resulting sequences are padded to a 
        maximum length corresponding to the longest sentence in the training
        set.
        """
        # List of questions
        self.train_questions = [dic["question"]
                     for dic in self.questions_train]
        self.test_questions = [dic["question"]
                     for dic in self.questions_test]

        # Create the tokenizer
        tokenizer = Tokenizer()

        # Fit the tokenizer on the training set questions
        tokenizer.fit_on_texts(self.train_questions)

        # Vocabulary size of the training set to be used for the embedding
        # layer of the model
        self.vocabulary_size_train = len(tokenizer.word_index.keys())

        # Let's use the embedding that has been fit on the training data
        train_sequences = tokenizer.texts_to_sequences(self.train_questions)
        test_sequences = tokenizer.texts_to_sequences(self.test_questions)
        
        # Longest sentence in the training set
        self.input_length = max([len(sequence) for sequence in train_sequences])

        # Pad the sequences to a maximum length
        self.train_sequences = pad_sequences(train_sequences, 
                                               maxlen=self.input_length, 
                                               padding="post")
        self.test_sequences = pad_sequences(test_sequences, 
                                               maxlen=self.input_length, 
                                               padding="post")
    def process_images(self, n=100):
        """Resize train and test images and prepocess them accordingly to VGG16
        input.
        
        Parameters
        ----------
        n : int, optional
            Number of images to process (for both the train and test sets).
        """
        # List of tuples (image_id, data_subtype)
        # Ex: (1, "train2014")
        train_image_ids = list(set((dic["image_id"], dic["data_subtype"]) 
                           for dic in self.questions_train))
        test_image_ids = list(set((dic["image_id"], dic["data_subtype"]) 
                           for dic in self.questions_test))

        self.train_images = self.process_images_(train_image_ids[:n])
        self.test_images = self.process_images_(test_image_ids[:n])

    def process_images_(self, image_ids):
        """Resize images and preprocess them accordingly to VGG16 input.
        
        Parameters
        ----------
        image_ids : list(tuple)
            Each tuple is (image_id, data_subtype). The image ids are unique.
        """
        # List to store the arrays (each array is an image 224x224x3)
        imgs = []

        for image_id in tqdm(image_ids):
            # Resize the image (VGG16 input)
            img = image.load_img(os.path.join(img_dir(self.dataDir, self.dataType, 
                image_id[1]), img_file(image_id[1], image_id[0])), target_size=(224, 224)) 
            # Convert the image to an array
            img = image.img_to_array(img)
            imgs.append(img)

        # Put the images together in a single array
        imgs = np.stack(imgs)

        # Preprocess the images (VGG16 input)
        imgs = preprocess_input(imgs)   

        # Duplicate the array for each image three times to correspond to the 
        # number of training questions
        imgs = np.repeat(imgs, 3, axis=0)

        return imgs

    def get_most_common_answer(self):
        """Get the most common answer for each question of the train and test
        sets.
        """
        self.train_answers = self.get_most_common_answer_(self.annotations_train)
        self.test_answers = self.get_most_common_answer_(self.annotations_test)

    def get_most_common_answer_(self, annotations):
        """Get the most common answer per question (among the 10 answers).
        
        Parameters
        ----------
        annotations : list(dict)
            Annotations from the VQA API.
        
        Returns
        -------
        list(str)
            List of answers. 
        """
        answers = [Counter((annotation["answers"][i]["answer"] 
                   for i in range(10))).most_common(1)[0][0] 
                   for annotation in annotations]
        return answers

    def get_top_answers(self, top_n=999):
        """Get the top_n answers from the train set (all 10 answers per 
        question are considered) and create dictionaries that map each top 
        answer to an index and vice-versa.
        
        Parameters
        ----------
        top_n : int, optional
            n most frequent answers from the train set.
        """
        # Attribute to be reused in self.encode_answers_
        self.top_n = top_n

        # All the answers from the train set (10*number of questions)
        answers_train = [annotation["answers"][i]["answer"] for i in range(10) 
        for annotation in self.annotations_train]

        # Top n answers
        top_answers_train = [answer for (answer, occ) in 
        Counter(self.answers_train).most_common(top_n)]

        # Dictionary mapping an index a top answer
        self.idx_to_answer = {i: answer for (i, answer) 
                              in zip(range(top_n), self.top_answers_train)}
        # Inverse dictionary mapping a top answer to an index
        self.answer_to_idx = {answer: i for (i, answer) 
                              in self.idx_to_answer.items()}

    def encode_answers(self):
        """Encode train and test answers and turn the indices into categorical 
        vectors.
        """
        self.train_answers_ind = self.encode_answers_(self.train_answers)
        self.test_answers_ind = self.encode_answers_(self.test_answers)
        self.train_answers_categorical = to_categorical(self.train_answers_ind)
        self.test_answers_categorical = to_categorical(self.test_answers_ind)

    def encode_answers_(self, answers):
        """Encode answers according to the top_n most frequent answers of the 
        training set (from 0 to top_n-1). The answer which is not found among the 
        top_n answers from the training set is encoded as top_n.
        
        Parameters
        ----------
        answers : list(str)
            List of answers to encode.
        
        Returns
        -------
        list(int)
            List of answers as indices (from 0 to top_n).
        """
        answers_ind = [self.answer_to_idx[answer]
                       if answer in self.answer_to_idx.keys() else self.top_n
                       for answer in answers]
        return answers_ind




