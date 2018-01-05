#!/usr/bin/env python
# -*- coding: utf-8 -*- 

"""This class implements the baseline text method of selecting the most frequent 
answer (in a training set) for the question type of the image/question pair 
to make predictions on.
"""

import os

from collections import Counter

from tools import ann_file, ques_file, qtypes_file, save_results
from vqa_api.PythonHelperTools.vqaTools.vqa import VQA

class qTypePrior():

    def __init__(self, dataDir, versionType="", taskType="OpenEnded", 
                 dataType="mscoco", dataSubTypesTrain = ["train2014"]):
        """     
        Parameters
        ----------
        dataDir : str
            Data directory path.
        versionType : str, optional
            "" for v1.0 dataset, "v2_" for v2.0 dataset.
        taskType : str, optional
            "OpenEnded" only for v2.0. "OpenEnded" or "MultipleChoice" for v1.0.
        dataType : str, optional
            "mscoco" only for v1.0. "mscoco" for real and "abstract_v002" for 
            abstract for v1.0. 
        dataSubTypeTrain : list or str, optional
            dataSubTypesTrain to train the method on. Can be a single string 
            ("train2014") or a list (["train2014", "val2014"]).
        """
        if type(dataSubTypesTrain) == list:
            dataSubTypesTrain = dataSubTypesTrain
        else:
            dataSubTypesTrain = [dataSubTypesTrain]

        self.dataDir = dataDir
        self.versionType = versionType
        self.taskType = taskType
        self.dataType = dataType

        # File names for the annotations and the questions
        for dataSubType in dataSubTypesTrain:
            setattr(self, "annFile_{}".format(dataSubType), ann_file(dataDir, 
                versionType, dataType, dataSubType))
            setattr(self, "quesFile_{}".format(dataSubType), ques_file(dataDir, 
                versionType, taskType, dataType, dataSubType))

        # Initialize VQA api for each dataSubType
        for dataSubType in dataSubTypesTrain:
            print("--> {}".format(dataSubType))
            setattr(self, "vqa_{}".format(dataSubType), 
                    VQA(getattr(self, "annFile_{}".format(dataSubType)), 
                        getattr(self, "quesFile_{}".format(dataSubType))))

        # Merge the annotations of the two different dataSubTypesTrain
        self.annotations = getattr(self, "vqa_{}".format(dataSubTypesTrain[0]))\
                           .dataset["annotations"]

        if len(dataSubTypesTrain) > 1:
            print("--> Merging the annotations of the different dataSubTypesTrain")
            for dataSubType in dataSubTypesTrain[1:]:
                self.annotations += getattr(self, "vqa_{}".format(dataSubType))\
                                    .dataset["annotations"]

        # Question types 
        with open(qtypes_file(dataDir, dataType), "r") as f:
            self.question_types = f.read().splitlines()

    def get_top_answer_per_qtype(self):
        """Get the most frequent answer per question type in the training data.
        """
        d = {q_type: [annotation["answers"][i]["answer"] for i in range(10) 
             for annotation in self.annotations 
             if annotation["question_type"] == q_type] 
             for q_type in self.question_types}

        self.qtype_to_top_answer = {q_type: 
                                    Counter(d[q_type]).most_common(1)[0][0] 
                                    for q_type in self.question_types}

    def predict(self, dataSubType):
        """Predict the answers for a dataSubType. The method loads the data 
        first.

        Parameters
        ----------
        dataSubType : str
            "val2014" for example.
        """
        # File names for the annotations and questions
        self.annFile_test = ann_file(self.dataDir, self.versionType, 
                                     self.dataType, dataSubType)
        self.quesFile_test = ques_file(self.dataDir, self.versionType, 
                                       self.taskType, self.dataType, dataSubType)

        # Initialize VQA api for the dataSubType on which to make predictions 
        # on the answers
        self.vqa_test = VQA(self.annFile_test, self.quesFile_test)

        question_ids_test = {annotation["question_id"]: annotation["question_type"] 
                             for annotation in  self.vqa_test.dataset["annotations"]}

        # Result list [{"answer": "most popular answer for the question type", 
        # "question_id": 1}] 
        res = [{"answer":self.qtype_to_top_answer[question_ids_test[question_id]], 
               "question_id": question_id} for question_id in question_ids_test]

        # Save the results
        self.results_file = save_results(res, self.dataDir, self.taskType, 
                                         self.dataType, dataSubType, 
                                         self.__class__.__name__)
        


        



