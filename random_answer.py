#!/usr/bin/env python
# -*- coding: utf-8 -*- 

"""This class implements the baseline text method of randomly selecting an 
answer among the top 1000 answer of the train dataset.
"""

import random
import json
import os

from vqa_api.PythonHelperTools.vqaTools.vqa import VQA
from tools import ann_file, ques_file, img_dir, save_results
from collections import Counter

class RandomAnswer():

	def __init__(self, dataDir, versionType="", taskType="OpenEnded", dataType="mscoco", 
		         dataSubTypes = ["train2014"]):

		self.dataDir = dataDir
		self.versionType = versionType
		self.taskType = taskType
		self.dataType = dataType

		# File names for the annotations and the questions
		for dataSubType in dataSubTypes:
			setattr(self, "annFile_{}".format(dataSubType), ann_file(dataDir, 
				versionType, dataType, dataSubType))
			setattr(self, "quesFile_{}".format(dataSubType), ques_file(dataDir, 
				versionType, taskType, dataType, dataSubType))

		# Initialize VQA api for each dataSubType
		for dataSubType in dataSubTypes:
			print "--> {}".format(dataSubType)
			setattr(self, "vqa_{}".format(dataSubType), 
				VQA(getattr(self, "annFile_{}".format(dataSubType)), getattr(self, "quesFile_{}".format(dataSubType))))

		# Merge the annotations of the two different dataSubTypes
		self.annotations = getattr(self, "vqa_{}".format(dataSubTypes[0])).dataset["annotations"]
		if len(dataSubTypes) > 1:
			print "--> Merging the annotations of the different dataSubTypes"
			for dataSubType in dataSubTypes[1:]:
				self.annotations += getattr(self, "vqa_{}".format(dataSubType)).dataset["annotations"]

	def get_top_answers(self, top_n=1000):
		self.answers = [annotation["answers"][i]["answer"] for i in range(10) 
		for annotation in self.annotations]
		self.top_answers = [answer for (answer, occ) in Counter(self.answers).most_common(1000)]

	def predict(self, dataSubType):

		# File names for the annotations and questions
		self.annFile_test = ann_file(self.dataDir, self.versionType, self.dataType, dataSubType)
		self.quesFile_test = ques_file(self.dataDir, self.versionType, self.taskType, self.dataType, dataSubType)

		# Initialize VQA api for the dataSubType on which to make predictions 
		# on the answers
		self.vqa_test = VQA(self.annFile_test, self.quesFile_test)

		question_ids_test = [annotation["question_id"] for annotation in  self.vqa_test.dataset["annotations"]]

		# Result list [{"answer": "no", "question_id": 1}] 
		res = [{"answer": random.choice(self.top_answers), "question_id": question_id} 
		for question_id in question_ids_test]

		# Save the results
		save_results(res, self.dataDir, self.taskType, self.dataType, dataSubType, 
			         self.__class__.__name__)
		



