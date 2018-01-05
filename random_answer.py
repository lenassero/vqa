#!/usr/bin/env python
# -*- coding: utf-8 -*- 

"""This class implements the baseline text method of randomly selecting an 
answer among the top 1000 answer of the train dataset.
"""

import random
import os

from collections import Counter

from vqa_api.PythonHelperTools.vqaTools.vqa import VQA
from tools import ann_file, ques_file, save_results

class RandomAnswer():

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
            dataSubTypes to train the method on. Can be a single string 
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
            print "--> {}".format(dataSubType)
            setattr(self, "vqa_{}".format(dataSubType), 
                VQA(getattr(self, "annFile_{}".format(dataSubType)), 
                    getattr(self, "quesFile_{}".format(dataSubType))))

        # Merge the annotations of the two different dataSubTypesTrain
        self.annotations = getattr(self, "vqa_{}".format(dataSubTypesTrain[0]))\
                           .dataset["annotations"]

        if len(dataSubTypesTrain) > 1:
            print "--> Merging the annotations of the different dataSubTypesTrain"
            for dataSubType in dataSubTypesTrain[1:]:
                self.annotations += getattr(self, "vqa_{}".format(dataSubType))\
                                    .dataset["annotations"]

    def get_top_answers(self, top_n=1000):
        """Get the top answers for the training data.
        
        Parameters
        ----------
        top_n : int, optional
            Number of answers to consider.
        """
        self.answers = [annotation["answers"][i]["answer"] for i in range(10) 
                        for annotation in self.annotations]
        self.top_answers = [answer for (answer, occ) 
                            in Counter(self.answers).most_common(1000)]

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

        question_ids_test = [annotation["question_id"] for annotation 
                             in self.vqa_test.dataset["annotations"]]

        # Result list [{"answer": "no", "question_id": 1}] 
        res = [{"answer": random.choice(self.top_answers), "question_id": question_id} 
               for question_id in question_ids_test]

        # Save the results
        self.results_file = save_results(res, self.dataDir, self.taskType, 
                                         self.dataType, dataSubType, 
                                         self.__class__.__name__)
        



