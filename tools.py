#!/usr/bin/env python
# -*- coding: utf-8 -*- 

"""Functions to use across the different methods.
"""

def ann_file(dataDir, versionType, dataType, dataSubType):
	filename = "{}/Annotations/{}{}_{}_annotations.json".format(dataDir, 
		versionType, dataType, dataSubType)
	return filename

def ques_file(dataDir, versionType, taskType, dataType, dataSubType):
	filename = "{}/Questions/{}{}_{}_{}_questions.json".format(dataDir, 
		versionType, taskType, dataType, dataSubType)
	return filename

def img_dir(dataDir, dataType, dataSubType):
	dirname = "{}/Images/{}/{}/".format(dataDir, dataType, dataSubType)
	return dirname