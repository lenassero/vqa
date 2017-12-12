#!/usr/bin/env python
# -*- coding: utf-8 -*- 

"""This class implements the baseline text method that always answers "yes".
"""

import random
import json
import os

from vqa_api.PythonHelperTools.vqaTools.vqa import VQA
from tools import ann_file, ques_file, img_dir
from collections import Counter


class PriorYesAnswer(object):

	def __init__(self, dataDir, versionType="", taskType="OpenEnded", dataType="mscoco"):

		self.dataDir = dataDir
		self.versionType = versionType
		self.taskType = taskType
		self.dataType = dataType
		
	def predict(self, dataSubType):

		# File names for the annotations and questions
		self.annFile_test = ann_file(self.dataDir, self.versionType, self.dataType, dataSubType)
		self.quesFile_test = ques_file(self.dataDir, self.versionType, self.taskType, self.dataType, dataSubType)

		# Initialize VQA api for the dataSubType on which to make predictions 
		# on the answers
		self.vqa_test = VQA(self.annFile_test, self.quesFile_test)

		question_ids_test = [annotation["question_id"] for annotation in  self.vqa_test.dataset["annotations"]]

		# Result list [{"answer": "yes", "question_id": 1}] 
		res = [{"answer": "yes", "question_id": question_id} 
		for question_id in question_ids_test]

		# Save the results
		save_results(res, self.dataDir, self.taskType, self.dataType, dataSubType, 
			         self.__class__.__name__)
		


		

