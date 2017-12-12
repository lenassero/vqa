#!/usr/bin/env python
# -*- coding: utf-8 -*- 

"""Functions to use across the different methods.
"""

import json
import os

def ann_file(dataDir, versionType, dataType, dataSubType):
	filename = os.path.join(dataDir, "Annotations", "{}{}_{}_annotations.json".\
		format(versionType, dataType, dataSubType))
	return filename

def ques_file(dataDir, versionType, taskType, dataType, dataSubType):
	filename = os.path.join(dataDir, "Questions", "{}{}_{}_{}_questions.json".\
		format(versionType, taskType, dataType, dataSubType))
	return filename

def img_dir(dataDir, dataType, dataSubType):
	dirname = os.path.join(dataDir, "Images", dataType, dataSubType)
	return dirname

def qtypes_file(dataDir, dataType="mscoco"):
	filename = os.path.join(dataDir, "QuestionTypes", "{}_question_types.txt".\
		format(dataType))
	return filename

def save_results(results, dataDir, taskType, dataType, dataSubType, methodName):
	"""Save the results obtained with the specific method in the Results 
	directory.
	
	Parameters
	----------
	results : list (dict)
	    List of dictionaries with the predicted answer for each question id :
	    [{"answer": "no", "question_id": 1}].
	dataDir : TYPE
	    Description
	taskType : TYPE
	    Description
	dataType : TYPE
	    Description
	dataSubType : TYPE
	    Description
	methodName : TYPE
	    Description
	"""
	print "--> Saving the results"

	results_file = os.path.join(dataDir, "Results", "{}_{}_{}_{}_results.json".\
		format(taskType, dataType, dataSubType, methodName))

	with open(results_file, "w") as f:
	    json.dump(results, f)

