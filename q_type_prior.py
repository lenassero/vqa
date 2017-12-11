#!/usr/bin/env python
# -*- coding: utf-8 -*- 

"""This class implements the baseline text method of selecting the most popular
answer for the identified question
"""

import random
import json
import os

from vqa_api.PythonHelperTools.vqaTools.vqa import VQA
from tools import ann_file, ques_file, img_dir
from collections import Counter

class qTypePrior():

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
			print("--> {}".format(dataSubType))
			setattr(self, "vqa_{}".format(dataSubType), 
				VQA(getattr(self, "annFile_{}".format(dataSubType)), getattr(self, "quesFile_{}".format(dataSubType))))

		# Merge the annotations of the two different dataSubTypes
		self.annotations = getattr(self, "vqa_{}".format(dataSubTypes[0])).dataset["annotations"]
		if len(dataSubTypes) > 1:
			print("--> Merging the annotations of the different dataSubTypes")
			for dataSubType in dataSubTypes[1:]:
				self.annotations += getattr(self, "vqa_{}".format(dataSubType)).dataset["annotations"]

	def get_top_answers(self):

		path_question_types = os.path.join(self.dataDir, "QuestionTypes")

		with open(os.path.join(path_question_types, "mscoco_question_types.txt"), "r") as f:
			question_types = f.read().splitlines()
		print(question_types)
		d = {q_type: [annotation["answers"][i]["answer"] for i in range(10) for annotation in self.annotations if annotation["question_type"] == q_type] for q_type in question_types}

        self.top_answers = {q_type: Counter(d[q_type]).most_common(1)[0][0] for q_type in question_types}



	def predict(self, dataSubType):

		# File names for the annotations and questions
		self.annFile_test = ann_file(self.dataDir, self.versionType, self.dataType, dataSubType)
		self.quesFile_test = ques_file(self.dataDir, self.versionType, self.taskType, self.dataType, dataSubType)

		# Initialize VQA api for the dataSubType on which to make predictions 
		# on the answers
		self.vqa_test = VQA(self.annFile_test, self.quesFile_test)

		question_ids_test = {annotation["question_id"]: annotation["question_type"]
		for annotation in  self.vqa_test.dataset["annotations"]}

		# Result list [{"answer": "most popular answer for the question type", "question_id": 1}] 
		res = [{"answer":self.top_answers[question_ids_test[question_id]], "question_id": question_id} 
		for question_id in question_ids_test]

		print("--> Saving the results")
		with open(os.path.join(
			self.dataDir, "Results/{}_{}_{}_{}_results.json".format(
				self.taskType, self.dataType, dataSubType, self.__class__.__name__)), "w") as f:
		    json.dump(res, f)


		



