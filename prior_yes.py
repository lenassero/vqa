#!/usr/bin/env python
# -*- coding: utf-8 -*- 

"""This class implements the baseline text method that always answers "yes".
"""

import os

from tools import ann_file, ques_file, img_dir, save_results
from vqa_api.PythonHelperTools.vqaTools.vqa import VQA

class PriorYes():

    def __init__(self, dataDir, versionType="", taskType="OpenEnded", 
                 dataType="mscoco"):
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
        """
        self.dataDir = dataDir
        self.versionType = versionType
        self.taskType = taskType
        self.dataType = dataType
        
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

        # Result list [{"answer": "yes", "question_id": 1}] 
        res = [{"answer": "yes", "question_id": question_id} 
               for question_id in question_ids_test]

        # Save the results
        self.results_file = save_results(res, self.dataDir, self.taskType, 
                                         self.dataType, dataSubType, 
                                         self.__class__.__name__)
        


        

