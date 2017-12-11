#!/usr/bin/env python
# -*- coding: utf-8 -*- 

"""Functions to use across the different methods.
"""

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
