#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Functions to use across the different methods.
"""

import json
import os


def ann_file(dataDir, versionType, dataType, dataSubType):
    filename = os.path.join(dataDir, "Annotations", "{}{}_{}_annotations.json".
                            format(versionType, dataType, dataSubType))
    return filename


def ques_file(dataDir, versionType, taskType, dataType, dataSubType):
    filename = os.path.join(dataDir, "Questions", "{}{}_{}_{}_questions.json".
                            format(versionType, taskType, dataType, dataSubType))
    return filename


def img_dir(dataDir, dataType, dataSubType):
    dirname = os.path.join(dataDir, "Images", dataType, dataSubType)
    return dirname


def qtypes_file(dataDir, dataType="mscoco"):
    filename = os.path.join(dataDir, "QuestionTypes", "{}_question_types.txt".
                            format(dataType))
    return filename

def img_file(dataSubType, imgId):
    filename = ".".join(("_".join(("COCO", dataSubType, str(imgId).zfill(12))), "jpg"))
    return filename

def skipthoughts_npy_file(dataDir, taskType, dataType, dataSubTypes, n):
    if type(dataSubTypes) == list:
        dataSubTypes = dataSubTypes
    else:
        dataSubTypes = [dataSubTypes]

    if not n:        
        filename = os.path.join(dataDir, "Embeddings",
                                "{}_{}_{}_skipthoughts.npy"
                                .format(taskType, dataType, "_".join(dataSubTypes)))
    else:
        filename = os.path.join(dataDir, "Embeddings",
                        "{}_{}_{}_skipthoughts_{}.npy"
                        .format(taskType, dataType, "_".join(dataSubTypes), n))

    return filename

def skipthoughts_idx_to_qid_file(dataDir, taskType, dataType, dataSubTypes, n):
    if type(dataSubTypes) == list:
        dataSubTypes = dataSubTypes
    else:
        dataSubTypes = [dataSubTypes]

    if not n:
        filename = os.path.join(dataDir, "Embeddings",
                                "{}_{}_{}_skipthoughts_idx_to_qid.pkl"
                                .format(taskType, dataType, "_".join(dataSubTypes)))
    else:
        filename = os.path.join(dataDir, "Embeddings",
                                "{}_{}_{}_skipthoughts_idx_to_qid_{}.pkl"
                                .format(taskType, dataType, "_".join(dataSubTypes), n))
    return filename

def skipthoughts_test_qid_to_train_knn_qids_file(dataDir, taskType, dataType, 
    dataSubTypesTrain, dataSubTypeTest, k, n_train, n_test):
    if type(dataSubTypesTrain) == list:
        dataSubTypesTrain = dataSubTypesTrain
    else:
        dataSubTypesTrain = [dataSubTypesTrain]

    if not n_train and not n_test :
        filename = os.path.join(dataDir, "Embeddings",
                                "{}_{}_skipthoughts_{}_qid_to_{}_{}nn_qids.pkl"
                                .format(taskType, dataType, dataSubTypeTest, 
                                    "_".join(dataSubTypesTrain), k))
    else:
        filename = os.path.join(dataDir, "Embeddings",
                                "{}_{}_skipthoughts_{}_{}_qid_to_{}_{}_{}nn_qids.pkl"
                                .format(taskType, dataType, dataSubTypeTest, 
                                    n_test, "_".join(dataSubTypesTrain), n_test, k))
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

    results_file = os.path.join(dataDir, "Results", "{}_{}_{}_{}_results.json".
                                format(taskType, dataType, dataSubType, methodName))

    with open(results_file, "w") as f:
        json.dump(results, f)
