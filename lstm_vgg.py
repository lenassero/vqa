#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import keras
import sys
import os

from vqa_api.PythonHelperTools.vqaTools.vqa import VQA
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing import image
from collections import Counter
from keras.models import Model
from tqdm import tqdm

from tools import ann_file, ques_file, img_dir, img_file, glove_dir

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

        # # Keep track of the questions ids
        # self.train_questions_ids = [dic["question_id"]
        #              for dic in self.questions_train]
        # self.test_questions_ids = [dic["question_id"]
        #              for dic in self.questions_test]

    def tokenize_questions(self):
        """ Train a tokenizer on the training set questions and tokenize the 
        train and set questions. The resulting sequences are padded to a 
        maximum length corresponding to the longest sentence in the training
        set.
        """
        # List of questions (we select only the questions in the training set
        # that correspond to answers belonging to the top 1000 answers of the 
        # training set)
        self.test_questions = [dic["question"]
                     for dic in self.questions_test]
        self.train_questions = [dic["question"]
                     for dic in self.questions_train]

        # Create the tokenizer
        self.tokenizer = Tokenizer()

        # Fit the tokenizer on the training set questions
        print("Fitting the tokenizer on the training questions ...")
        self.tokenizer.fit_on_texts(self.train_questions)

        # Vocabulary size of the training set to be used for the embedding
        # layer of the model
        self.vocabulary_size_train = len(self.tokenizer.word_index.keys())

        # Let's use the embedding that has been fit on the training data
        print("Embedding the train and test questions ...")
        train_sequences = self.tokenizer.texts_to_sequences(self.train_questions)
        test_sequences = self.tokenizer.texts_to_sequences(self.test_questions)
        
        # Longest sentence in the training set
        self.input_length = max([len(sequence) for sequence in train_sequences])

        # Pad the sequences to a maximum length
        print("Padding the sequences to a maximum length of {} ...".\
              format(self.input_length))
        self.train_sequences = pad_sequences(train_sequences, 
                                               maxlen=self.input_length, 
                                               padding="post")
        self.test_sequences = pad_sequences(test_sequences, 
                                               maxlen=self.input_length, 
                                               padding="post")

    def create_embedding_matrix(self, embedding_dim=300):
        """Create an embedding matrix of the training set vocabulary using Glove 
        pre-trained embeddings.
        
        Parameters
        ----------
        embedding_dim : int, optional
            Embedding dimension of the word vectors.
        """
        embedding_matrix_filename = os.path.join(glove_dir(self.dataDir), "embedding_matrix.npy")
        if os.path.exists(embedding_matrix_filename):
            self.embedding_matrix = np.load(embedding_matrix_filename)
        else:            
            # Load Glove embeddings (Wikipedia 2014 + Gigaword 5) and map each word
            # to its embedding
            embeddings_index = {}
            f = open(os.path.join(glove_dir(self.dataDir), "glove.6B.{}d.txt".format(embedding_dim)))
            print("Reading Glove embedding and creating an embedding index {word: embedding array} ...")
            for line in tqdm(f):
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype="float32")
                embeddings_index[word] = coefs
            f.close()
            print("--> Found {} word vectors in Glove embedding.".format(len(embeddings_index)))

            # Create the embedding matrix mapping each word in the vocabulary to 
            # each embedding vector found in Glove (0 if not)
            print("\n")
            print("Creating the embedding matrix ...")
            self.embedding_matrix = np.zeros((self.vocabulary_size_train + 1, embedding_dim))
            for word, i in tqdm(self.tokenizer.word_index.items()):
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    # Words not found in embedding index will be all-zeros
                    self.embedding_matrix[i] = embedding_vector

            # Save the embedding matrix
            np.save(embedding_matrix_filename, self.embedding_matrix)
            
    def process_images_train(self, n=100):
        """Resize train and test images and prepocess them accordingly to VGG16
        input.
        
        Parameters
        ----------
        n : int, optional
            Number of images to process (for both the train and test sets).
        """
        # List of tuples (image_id, data_subtype)
        # Ex: (1, "train2014")
        # We select only images in the training set that correspond to answers 
        # belonging to the top 1000 answers of the training set
        train_image_ids = [(dic["image_id"], dic["data_subtype"]) 
                           for dic in self.questions_train]

        # Each image id is replicated three times (3 questions), let's reduce 
        # the size of the above lists, to avoid encoding the same image three
        # times
        # train_image_ids = [train_image_ids[i] 
        # for i in range(len(train_image_ids)) if i%3 == 0]
        # test_image_ids = [test_image_ids[i] 
        # for i in range(len(test_image_ids)) if i%3 == 0]

        self.train_images = self.process_images_(train_image_ids[:n])

    def process_encode_images_train(self, n=100):
        """Resize train and test images and prepocess them accordingly to VGG16
        input.
        
        Parameters
        ----------
        n : int, optional
            Number of images to process (for both the train and test sets).
        """
        train_image_ids = [(dic["image_id"], dic["data_subtype"]) 
                           for dic in self.questions_train]

        self.train_images = self.process_encode_images_(train_image_ids[:n])

    def process_images_test(self, n=100):
        """Resize train and test images and prepocess them accordingly to VGG16
        input.
        
        Parameters
        ----------
        n : int, optional
            Number of images to process (for both the train and test sets).
        """
        # List of tuples (image_id, data_subtype)
        # Ex: (1, "train2014")
        # We select only images in the training set that correspond to answers 
        # belonging to the top 1000 answers of the training set
        test_image_ids = [(dic["image_id"], dic["data_subtype"]) 
                           for dic in self.questions_test]
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
        # List of images that have been processed {image_id: idx}
        image_ids_processed = {}
        # Keep track of the first index corresponding to the first time an 
        # image has been processed
        i = 0

        for image_id in tqdm(image_ids):
            if image_id[0] in image_ids_processed.keys():
                img = imgs[image_ids_processed[image_id[0]]]
            else:
                # Resize the image (VGG16 input)
                img = image.load_img(os.path.join(img_dir(self.dataDir, self.dataType, 
                    image_id[1]), img_file(image_id[1], image_id[0])), target_size=(224, 224)) 
                # Convert the image to an array
                img = image.img_to_array(img)
                # Increment i if the image has not been processed yet
                image_ids_processed[image_id[0]] = i
            i += 1
            imgs.append(img)

        # Put the images together in a single array
        imgs = np.stack(imgs)

        # Preprocess the images (VGG16 input)
        imgs = preprocess_input(imgs)   

        # Duplicate the array for each image three times to correspond to the 
        # number of training questions
        # imgs = np.repeat(imgs, 3, axis=0)

        return imgs

    def process_encode_images_(self, image_ids):
        """Resize images and preprocess them accordingly to VGG16 input.
        
        Parameters
        ----------
        image_ids : list(tuple)
            Each tuple is (image_id, data_subtype). The image ids are unique.
        """
        vgg_model = keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', 
                               input_tensor=None, input_shape=None, 
                               pooling=None, classes=1000)

        # "Submodel" of VGG until the fc7 layer
        vgg_model_fc7 = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer("fc2").output)

        # List to store the arrays (each array is a vector 4096)
        embeddings = []
        # List of images that have been processed {image_id: idx}
        image_ids_processed = {}
        # Keep track of the first index corresponding to the first time an 
        # image has been processed
        i = 0

        for image_id in tqdm(image_ids):
            if image_id[0] in image_ids_processed.keys():
                img = embeddings[image_ids_processed[image_id[0]]]
            else:
                # Resize the image (VGG16 input)
                img = image.load_img(os.path.join(img_dir(self.dataDir, self.dataType, 
                    image_id[1]), img_file(image_id[1], image_id[0])), target_size=(224, 224)) 
                # Convert the image to an array
                img = image.img_to_array(img)
                # Preprocess the image (VGG16 input)
                img = preprocess_input(img)
                # Reshape the image
                img = img.reshape(1, 224, 224, 3)
                # Get the embedding for the current image
                embedding = vgg_model_fc7.predict(img)
                # Turn the embedding to a vector 
                embedding = embedding.flatten()
                # Increment i if the image has not been processed yet
                image_ids_processed[image_id[0]] = i
            i += 1
            embeddings.append(embedding)

        # Put the images together in a single array
        embeddings = np.stack(embeddings) 

        return embeddings

    def get_most_common_answer(self):
        """Get the most common answer for each question of the train and test
        sets.
        """
        print("Getting the ground truth answer for each question/image in the"
              " training and test sets ...")
        # For the training set, we select only the answers belonging to the 
        # top 1000 answers
        self.train_answers = self.get_most_common_answer_(self.annotations_train)
        self.test_answers = self.get_most_common_answer_(self.annotations_test)

        # Add the ground truth answers in the questions dictionaries
        for i in range(len(self.questions_train)):
            self.questions_train[i]["answer"] = self.train_answers[i]
        for i in range(len(self.questions_test)):
            self.questions_test[i]["answer"] = self.test_answers[i]

    def get_most_common_answer_(self, annotations):
        """Get the most common answer per question (among the 10 answers).
        
        Parameters
        ----------
        annotations : list(dict)
            Annotations from the VQA API.
        
        Returns
        -------
        list(str)
            List of answers (question_id, answer). 
        """
        answers = [Counter((annotation["answers"][i]["answer"] 
                   for i in range(10))).most_common(1)[0][0]
                   for annotation in annotations]
        return answers

    def get_top_answers(self, top_n=1000):
        """Get the top_n answers from the train set (all 10 answers per 
        question are considered) and create dictionaries that map each top 
        answer to an index and vice-versa.
        
        Parameters
        ----------
        top_n : int, optional
            n most frequent answers from the train set.
        """
        print("Getting the top {} answers from the training set ...".format(top_n))
        # Attribute to be reused in self.encode_answers_
        self.top_n = top_n

        # All the answers from the train set (10*number of questions)
        answers_train = [annotation["answers"][i]["answer"] for i in range(10) 
        for annotation in self.annotations_train]

        # Top n answers
        top_answers_train = [answer for (answer, occ) in 
        Counter(answers_train).most_common(top_n)]

        print("Mapping each top answer to an index between 0 and {} in"
              " self.idx_to_answer_dic and self.answer_to_idx_dic ...".format(top_n-1))
        # Dictionary mapping an index to a top answer
        self.idx_to_answer_dic = {i: answer for (i, answer) 
                              in zip(range(top_n), top_answers_train)}
        # Inverse dictionary mapping a top answer to an index
        self.answer_to_idx_dic = {answer: i for (i, answer) 
                              in self.idx_to_answer_dic.items()}

    def reduce_train_qids(self):
        """Reduce the training set question ids to the ones which are among the 
        top 1000 answers of the training set.
        """
        self.train_questions_ids = [question_id for (question_id, answer) 
        in zip(self.train_questions_ids, self.train_answers) 
        if answer in self.answer_to_idx_dic.keys()]

    def reduce_train_answers(self):
        """Reduce the training set answers to the ones which are among the 
        top 1000 answers of the training set.
        """
        print("Keeping only every question/image with an answer among the top {} answers"
              " of the training set ...".format(self.top_n))
        self.train_answers = [answer for answer in self.train_answers if answer
                              in self.answer_to_idx_dic.keys()]
        self.questions_train = [question for question in self.questions_train 
                                if question["answer"] in self.answer_to_idx_dic.keys()]

    def encode_answers(self):
        """Encode train answers and turn the indices into categorical vectors.
        """
        print("Encoding train and test answers ...")
        self.train_answers_ind = self.encode_answers_(self.train_answers)
        self.train_answers_categorical = to_categorical(self.train_answers_ind)
        # self.test_answers_ind = self.encode_answers_(self.test_answers)
        # self.test_answers_categorical = to_categorical(self.test_answers_ind)

    def encode_answers_(self, answers):
        """Encode answers according to the top_n most frequent answers of the 
        training set (from 0 to top_n-1).
        
        Parameters
        ----------
        answers : list(str)
            List of answers to encode.
        
        Returns
        -------
        list(int)
            List of answers as indices (from 0 to top_n-1).
        """
        answers_ind = [self.answer_to_idx_dic[answer] for answer in answers]
        return answers_ind

    def idx_to_answer(self, idx):
        """Return the answer corresponding to an index (from 0 to 999).
        Parameters
        ----------
        idx : int
            Index of the answer resulting from the encoding of the answers.
        
        Returns
        -------
        str
            Answer.
        """
        return self.idx_to_answer_dic[idx]

    def clear_variables(self):
        """Delete variables after processing the data.
        """

        self.delete_variable(self.questions_train)
        self.delete_variable(self.questions_test)

        self.delete_variable(self.annotations_train)
        self.delete_variable(self.annotations_test)

        for dataSubType in self.dataSubTypesTrain:
            self.delete_variable(getattr(self, "vqa_{}".format(dataSubType)))
        self.delete_variable(self.vqa_test)

        self.delete_variable(self.train_answers)
        self.delete_variable(self.test_answers)

        self.delete_variable(self.train_questions)
        self.delete_variable(self.test_questions)

        self.delete_variable(self.tokenizer)

    def delete_variable(self, variable):
        try:
            del variable
        except:
            pass

    def predictions_to_dic(self, predictions, question_ids):
        """Turn the predictions from the model to the a result list as 
        required by the evaluation tool.
        
        Parameters
        ----------
        predictions : array, shape = [n, 1000]
            Array resulting from the predict function, n is the number of 
            question/image pair for which we have predicted an answer.
        question_ids : list(int)
            Question ids corresponding to the questions on which we make 
            predictions.
        
        Returns
        -------
        TYPE
            Description
        """
        # Predicted answers (as indices)
        answers_ind_pred = np.argmax(predictions, axis=1)

        # Predicted answers (as strings)
        idx_to_answer = np.vectorize(self.idx_to_answer)
        answers_pred = idx_to_answer(answers_ind_pred)

        # Result list [{"answer": "yes", "question_id": 1}] 
        res = [{"answer": answer, "question_id": question_id} 
        for (answer, question_id) in zip(answers_pred, question_ids)]

        # Save the results
        # save_results(res, self.dataDir, self.taskType, self.dataType, dataSubType, 
        #              self.__class__.__name__)

        return res


