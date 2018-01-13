#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This class implement the nearest neighbors method: for every question in the 
test set, the goal is to find the k nearest neighbors in the training set using 
cosine similarity after encoding the questions in skip-thought feature space. 
Then, in this set of k questions and their associated images, we find the image 
which is most similar to the query image using cosine similarity after encoding 
the images with the fc7 features from the VGG16 model.
"""
import cPickle as pkl
import numpy as np
import keras
import sys
import os

from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from collections import Counter
from keras.models import Model
from tqdm import tqdm

from tools import ann_file, img_file, ques_file, img_dir,\
                  skipthoughts_npy_file,\
                  skipthoughts_idx_to_qid_file,\
                  skipthoughts_test_qid_to_train_knn_qids_file,\
                  save_results, vgg_embeddings_nn_file
from vqa_api.PythonHelperTools.vqaTools.vqa import VQA
from utils import chunking_dot
# from skip_thoughts import skipthoughts


class NearestNeighbors():

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

        # Initialize VQA api for each dataSubType
        for dataSubType in dataSubTypesTrain:
            print("---> {}".format(dataSubType))
            setattr(self, "vqa_{}".format(dataSubType),
                    VQA(getattr(self, "annFile_{}".format(dataSubType)),
                        getattr(self, "quesFile_{}".format(dataSubType))))

        # Merge the questions of the two different dataSubTypesTrain
        self.questions_train = getattr(
            self, "vqa_{}".format(dataSubTypesTrain[0])).questions["questions"]
        for dic in self.questions_train:
            dic["data_subtype"] = dataSubTypesTrain[0]

        if len(dataSubTypesTrain) > 1:
            print("--> Merging the annotations of the different dataSubTypesTrain")
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

        # File names for the annotations and questions
        self.annFile_test = ann_file(
            self.dataDir, self.versionType, self.dataType, self.dataSubTypeTest)
        self.quesFile_test = ques_file(
            self.dataDir, self.versionType, self.taskType, self.dataType, self.dataSubTypeTest)

        # Initialize VQA api for the self.dataSubTypeTest on which to make predictions
        # on the answers
        print("\n")
        print("---> {}".format(dataSubTypeTest))
        self.vqa_test = VQA(self.annFile_test, self.quesFile_test)
        self.questions_test = self.vqa_test.questions["questions"]
        for dic in self.questions_test:
            dic["data_subtype"] = self.dataSubTypeTest

        # Reduce the size of the test set
        self.questions_test = self.questions_test[:n_test]
        self.n_test = n_test

        # Load the skipthoughts model and initialize the encoder
        print("\n")
        print("---> Loading the skipthoughts model ...")
        # self.model = skipthoughts.load_model()
        # self.encoder = skipthoughts.Encoder(self.model)

        # Load vggnet fc7 layer
        print("\n")
        print("---> Loading the VGG16 model ...")
        vgg_model = keras.applications.vgg16.VGG16(include_top=True, weights='imagenet',
                               input_tensor=None, input_shape=None,
                               pooling=None, classes=1000)
        self.fc7 = Model(inputs=vgg_model.input,
                         outputs=vgg_model.get_layer("fc2").output)

    def encode_questions_train(self):
        """Encode questions from the training set using skipthoughts.
        """

        # File name to save the array of vectors
        vectors_train_npy_file = skipthoughts_npy_file(
            self.dataDir, self.taskType, self.dataType, self.dataSubTypesTrain, self.n_train)
        vectors_train_idx_to_qid_file = skipthoughts_idx_to_qid_file(
            self.dataDir, self.taskType, self.dataType, self.dataSubTypesTrain, self.n_train)

        # Load the encoded vectors if they have already been encoded
        if os.path.exists(vectors_train_npy_file) and os.path.exists(vectors_train_idx_to_qid_file):
            print("The train questions have already been embedded, loading" 
                  " them ...")
            self.vectors_train = np.load(vectors_train_npy_file)
            with open(vectors_train_idx_to_qid_file, "r") as f:
                self.vectors_train_idx_to_qid = pkl.load(f)

        else:
            print("Encoding ...")
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
            print("Saving ...")
            with open(vectors_train_idx_to_qid_file, "w") as f:
                pkl.dump(self.vectors_train_idx_to_qid, f)
            print("Saved file: vectors_train_idx_to_qid_file")

    def encode_questions_test(self):
        """Encode questions from the test set using skipthoughts.
        """
        # File name to save the array of vectors
        vectors_test_npy_file = skipthoughts_npy_file(
            self.dataDir, self.taskType, self.dataType, self.dataSubTypeTest, self.n_test)
        vectors_test_idx_to_qid_file = skipthoughts_idx_to_qid_file(
            self.dataDir, self.taskType, self.dataType, self.dataSubTypeTest, self.n_test)

        # Load the encoded vectors if they have already been encoded
        if os.path.exists(vectors_test_npy_file) and os.path.exists(vectors_test_idx_to_qid_file):
            print("The test questions have already been embedded, loading" 
                  " them ...")
            self.vectors_test = np.load(vectors_test_npy_file)
            with open(vectors_test_idx_to_qid_file, "r") as f:

                self.vectors_test_idx_to_qid = pkl.load(f)

        else:
            print("Encoding ...")
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
            print("Saving the results ...")
            with open(vectors_test_idx_to_qid_file, "w") as f:
                pkl.dump(self.vectors_test_idx_to_qid, f)
            print("Saved file: {}".format(vectors_test_idx_to_qid_file))


    def compute_cosine_similarity(self, test_vectors, train_vectors):
        """Compute the cosine similarity between test embeddings and train 
        embeddings.
        
        Parameters
        ----------
        test_vectors : array, shape = [n_test, embedding_dim]
            n_test is the number of vectors, embedding_dim the embedding dimension
            (4800 in skipthoughts).
        train_vectors : array, shape = [n_train, embedding_dim]
            n_train is the number of vectors, embedding_dim the embedding dimension
            (4800 in skipthoughts).
        Returns
        -------
        array, shape = [n_test, n_train]
            Cosine similarities between each test vector and all the other train
            vectors.
        """
        test_norms = np.linalg.norm(test_vectors, axis=1)
        train_norms = np.linalg.norm(train_vectors, axis=1)

        cosine_similarities = test_vectors.dot(train_vectors.T)\
                              * (1/np.reshape(test_norms, (test_norms.shape[0], 1)))\
                              * (1/np.reshape(train_norms, (train_norms.shape[0], 1))).T

        return cosine_similarities

    def get_test_questions_nn_train(self, k=4):
        """For each question in the test set, get the k most similar questions
        in the training set using cosine similarities.
        
        Parameters
        ----------
        k : int, optional
            k nearest questions from the training set to get.
        """
        # File name to save the results
        test_qid_to_nn_train_qids_file = skipthoughts_test_qid_to_train_knn_qids_file(
            self.dataDir, self.taskType, self.dataType, self.dataSubTypesTrain,
            self.dataSubTypeTest, k, self.n_train, self.n_test)

        # Load the results if they exist
        if os.path.exists(test_qid_to_nn_train_qids_file):
            with open(test_qid_to_nn_train_qids_file, "r") as f:
                print("The test questions nearest neighbors in the train set"
                      " have already been computed, loading ...")
                self.test_qid_to_nn_train_qids = pkl.load(f)

        else:
            print("Getting the {} nearest train questions to each test"
                  " question ...".format(k))
            # Compute the cosine similarity between each test vector and each train
            # vector
            cs = self.compute_cosine_similarity(
                self.vectors_test, self.vectors_train, low_memory=low_memory, chunk_size=chunk_size)

            # Sort the scores in ascending order and return the indices
            cs_sorted_args = np.flip(np.argsort(cs, axis=1), 1)[:, :k]

            # Free memory
            del cs

            # Convert the train array indices to the question ids
            train_idx_to_qid = np.vectorize(
                lambda idx: self.vectors_train_idx_to_qid[idx])
            cs_sorted_args = train_idx_to_qid(cs_sorted_args)

            # Dictionnary mapping the each question id in the test set to the k
            # most similar questions (question ids) in the training set. The latters
            # are stored in an array
            self.test_qid_to_nn_train_qids = {self.vectors_test_idx_to_qid[idx]:
                                         cs_sorted_args[idx]
                                         for idx in range(cs_sorted_args.shape[0])}

            print("Saving the results ...")
            with open(test_qid_to_nn_train_qids_file, "w") as f:    
                pkl.dump(self.test_qid_to_nn_train_qids, f)
                print("Save file: {}".format(test_qid_to_nn_train_qids_file))

    def get_test_questions_nn_train_low_memory(self, k=4, chunk_size=5000):
        """For each question in the test set, get the k most similar questions
        in the training set using cosine similarities. This is the "low memory"
        version of the function get_test_questions_nn_train when there is not 
        enough memory to compute the dot product between all test embeddings
        and all train embeddings.
        
        Parameters
        ----------
        k : int, optional
            k nearest questions from the training set to get.
        chunk_size : int, optional
            The cosine similarities between test questions and train questions
            are computed using a chunk_size number of test questions at each
            iteration.
        """
        # File name to save the results
        test_qid_to_nn_train_qids_file = skipthoughts_test_qid_to_train_knn_qids_file(
            self.dataDir, self.taskType, self.dataType, self.dataSubTypesTrain,
            self.dataSubTypeTest, k, self.n_train, self.n_test)

        # Load the results if they exist    
        if os.path.exists(test_qid_to_nn_train_qids_file):
            with open(test_qid_to_nn_train_qids_file, "r") as f:
                print("The test questions nearest neighbors in the train set"
                      " have already been computed, loading ...")
                self.test_qid_to_nn_train_qids = pkl.load(f)
            # Free memory
            del self.vectors_train
            del self.vectors_test
        
        else: 
            # Norm of each train vector
            train_norms = np.linalg.norm(self.vectors_train, axis=1)

            # Number of train and test data
            n_test = self.vectors_test.shape[0]
            n_train = self.vectors_train.shape[0]

            # Function mapping the index of the train question (in the list) to 
            # the question id
            train_idx_to_qid = np.vectorize(
                lambda idx: self.vectors_train_idx_to_qid[idx])

            # Dictionnary mapping the each question id in the test set to the k
            # most similar questions (question ids) in the training set. The latters
            # are stored in an array
            self.test_qid_to_nn_train_qids = {}

            # Used to make indices and question ids correspond when computing 
            # with chunks
            offset = 0

            for i in tqdm(range(0, n_test, chunk_size)):
                test_norms = np.linalg.norm(self.vectors_test[i:i+chunk_size], axis=1)
                cosine_similarities_ = self.vectors_test[i:i+chunk_size].dot(self.vectors_train.T)\
                                            * (1/np.reshape(test_norms, (test_norms.shape[0], 1)))\
                                            * (1/np.reshape(train_norms, (train_norms.shape[0], 1))).T

                # Sort the scores in ascending order and return the indices
                cs_sorted_args = np.flip(np.argsort(cosine_similarities_, axis=1), 1)[:, :k]

                # Convert the train array indices to the question ids
                cs_sorted_args = train_idx_to_qid(cs_sorted_args)

                for idx in range(cs_sorted_args.shape[0]):
                    self.test_qid_to_nn_train_qids[self.vectors_test_idx_to_qid[idx+offset]] = cs_sorted_args[idx]

                offset += chunk_size

            print("Saving the results ...")
            with open(test_qid_to_nn_train_qids_file, "w") as f:    
                pkl.dump(self.test_qid_to_nn_train_qids, f)
                print("Save file: {}".format(test_qid_to_nn_train_qids_file))

    def encode_images_train(self):
        """Resize train images, prepocess them accordingly to VGG16 input, and 
        encode them to 4096 vectors with VGG16.
        """
        # Embedded images file (.pkl)
        vgg_embedding_filename = vgg_embeddings_nn_file(
            self.dataDir, self.taskType, self.dataType, self.dataSubTypesTrain)

        # Load the VGG embeddings if they have been created
        if os.path.exists(vgg_embedding_filename):
            print("The train images have already been encoded, loading the"
                  " embeddings ... ")
            with open(vgg_embedding_filename, "r") as f:
                self.fc7_train = pkl.load(f)

        else:
            train_image_ids = [(dic["image_id"], dic["data_subtype"]) 
                               for dic in self.questions_train]

            print("Encoding train images ...")
            self.fc7_train = self.encode_images(train_image_ids)

            print("Saving the results ...")
            with open(vgg_embedding_filename, "w") as f:
                pkl.dump(self.fc7_train, f)
                print("Saved file: {}".format(vgg_embedding_filename))

    def encode_images_test(self):
        """Resize test images, prepocess them accordingly to VGG16 input, and 
        encode them to 4096 vectors with VGG16.
        """
        # Embedded images file (.pkl)
        vgg_embedding_filename = vgg_embeddings_nn_file(
            self.dataDir, self.taskType, self.dataType, self.dataSubTypeTest)

        # Load the VGG embeddings if they have been created
        if os.path.exists(vgg_embedding_filename):
            print("The train images have already been encoded, loading the"
                  " embeddings ... ")
            with open(vgg_embedding_filename, "r") as f:
                self.fc7_test = pkl.load(f)

        else:
            test_image_ids = [(dic["image_id"], dic["data_subtype"]) 
                               for dic in self.questions_test]

            print("Encoding test images ...")
            self.fc7_test = self.encode_images(test_image_ids)

            print("Saving the results ...")
            with open(vgg_embedding_filename, "w") as f:
                pkl.dump(self.fc7_test, f)
                print("Saved file: {}".format(vgg_embedding_filename))

    def encode_images(self, image_ids):
        """Resize images, preprocess them accordingly to VGG16 input, and 
        encode them to 4096 vectors with VGG16. This is not a vectorized version
        (but memory efficient) since each image is loaded and encoded one by
        one. 
        # TO DO: implement the vectorized version (with chunks)
        
        Parameters
        ----------
        image_ids : list(tuple)
            Each tuple is (image_id, data_subtype). The image ids are unique.
        """
        # List to store the encodings 
        encoded_imgs = []

        image_ids_sliced = image_ids[::3]

        for image_id in tqdm(image_ids_sliced):

            # Resize the images as VGG inputs
            img = image.load_img(os.path.join(img_dir(self.dataDir, self.dataType,
                image_id[1]), img_file(image_id[1], image_id[0])), target_size=(224, 224))

            # Convert the image to an array
            img = image.img_to_array(img)

            # Preprocess the image (VGG16 input)
            img = preprocess_input(img)

            # Reshape the image
            img = img.reshape(1, 224, 224, 3)

            # Get the embedding for the current image
            encoded_img = self.fc7.predict(img)

            # Turn the embedding to a vector 
            encoded_img = encoded_img.flatten()

            encoded_imgs.append(encoded_img)

        encoded_imgs = np.stack(encoded_imgs)

        # Dictionary mapping image_id to the fc7 encoding of the image in the
        # ing
        fc7 = {image_ids_sliced[i][0]: encoded_imgs[i, ] 
               for i in range(np.shape(encoded_imgs)[0])}

        return fc7

    def get_test_images_nn_train(self):
        """For each test pair image/question for which k neareset train pairs 
        image/question have been associated by computing the cosine similarities
        between questions, get the most similar image from the train.
        """
        print("Getting the nearest image among the 4 train question/image nearest" 
              " pairs to each test question/image pair ...")
        # Dictionaries mapping question_id to image_id in train & test
        train_qID_to_imgID = {dic["question_id"]: dic["image_id"] for dic in self.questions_train}
        test_qID_to_imgID = {dic["question_id"]: dic["image_id"] for dic in self.questions_test}

        # Dictionary mapping (question_id, image_id) in test to the array of image_id
        # in train
        dic_imgID_nn = {(qID, test_qID_to_imgID[qID]): np.array([train_qID_to_imgID[q] 
                        for q in self.test_qid_to_nn_train_qids[qID]])
                        for qID in self.test_qid_to_nn_train_qids.keys()}

        # Dictionary mapping question_id from test to (fc7 from test, array of fc7 features from train)
        fc7_dic = {q[0]: 
                   (self.fc7_test[q[1]], np.array([self.fc7_train[e] for e in dic_imgID_nn[q]]))
                   for q in dic_imgID_nn.keys()}

        # Dictionary of cosine similarities
        cs_dic = {qID: 
                  np.divide(
                            fc7_dic[qID][0].dot(fc7_dic[qID][1].T), 
                            np.linalg.norm(fc7_dic[qID][0])*np.linalg.norm(fc7_dic[qID][1], axis=1)
                            )
                  for qID in fc7_dic.keys()}

        # Dictionary mapping question_id from test to index in the array of knn questions
        self.qID_to_img_nn_index = {qID: np.argmax(cs_dic[qID])
                                    for qID in cs_dic.keys()}

        # Dictionary mapping question_id from test to top_image in train (for sanity checks)
        self.qID_to_img_nn = {qID: dic_imgID_nn[(qID,test_qID_to_imgID[qID])][np.argmax(cs_dic[qID])]
                              for qID in cs_dic.keys()}

        # Dictionary mapping question_id from test to the sorted list of NN img in train (for sanity checks)
        # self.sorted_qID_to_img_nn = {qID: dic_imgID_nn[(qID,test_qID_to_imgID[qID])][np.argsort(cs_dic[qID])]
        #                              for qID in cs_dic.keys()}

    def predict(self, dataSubType):
        """Predict the answers for a dataSubType.

        Parameters
        ----------
        dataSubType : str
            "val2014" for example.
        """
        train_qID_to_imgID = {dic["question_id"]: dic["image_id"] 
                              for dic in self.questions_train}

        # Dicionary mapping q_id in train to its answer
        answers_train = {annotation["question_id"]: Counter((annotation["answers"][i]["answer"]
                         for i in range(10))).most_common(1)[0][0] 
                         for annotation in self.vqa_train2014.dataset["annotations"]}

        # Result list [{"answer": "picked from NN image", "question_id": 1}] 
        res = [{"answer": 
                answers_train[self.test_qid_to_nn_train_qids[q_id][self.qID_to_img_nn_index[q_id]]], 
                "question_id": q_id} 
               for q_id in self.qID_to_img_nn_index.keys()]

        # Save the results
        self.results_file = save_results(res, self.dataDir, self.taskType, 
                                         self.dataType, dataSubType, 
                                         self.__class__.__name__)
        









