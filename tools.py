#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Functions to use across the different methods.
"""

import json
import os

def ann_file(dataDir, versionType, dataType, dataSubType):
    """Annotations file path.
    
    Parameters
    ----------
    dataDir : str
        Data directory path.    
    versionType : str
        "" for v1.0 dataset, "v2_" for v2.0 dataset.
    dataType : str
        "mscoco" only for v1.0. "mscoco" for real and "abstract_v002" for 
        abstract for v1.0. 
    dataSubType : str
        "train2014" or "val2014" for example.
    
    Returns
    -------
    str
    """
    filename = os.path.join(dataDir, "Annotations", "{}{}_{}_annotations.json".
                            format(versionType, dataType, dataSubType))
    return filename


def ques_file(dataDir, versionType, taskType, dataType, dataSubType):
    """Questions file path.
    
    Parameters
    ----------
    dataDir : str
        Data directory path.    
    versionType : str
        "" for v1.0 dataset, "v2_" for v2.0 dataset.
    taskType : str
        "OpenEnded" only for v2.0. "OpenEnded" or "MultipleChoice" for v1.0.
    dataType : str
        "mscoco" only for v1.0. "mscoco" for real and "abstract_v002" for 
        abstract for v1.0. 
    dataSubType : str
        "train2014" or "val2014" for example.
    
    Returns
    -------
    str
    """
    filename = os.path.join(dataDir, "Questions", "{}{}_{}_{}_questions.json".
                            format(versionType, taskType, dataType, dataSubType))
    return filename


def img_dir(dataDir, dataType, dataSubType):
    """Images directory path.
    
    Parameters
    ----------
    dataDir : str
        Data directory path.    
    dataType : str
        "mscoco" only for v1.0. "mscoco" for real and "abstract_v002" for 
        abstract for v1.0. 
    dataSubType : str
        "train2014" or "val2014" for example.
    
    Returns
    -------
    str
    """
    dirname = os.path.join(dataDir, "Images", dataType, dataSubType)
    return dirname


def qtypes_file(dataDir, dataType="mscoco"):
    """Question types file path.
    
    Parameters
    ----------
    dataDir : str
        Data directory path.    
    dataType : str
        "mscoco" only for v1.0. "mscoco" for real and "abstract_v002" for 
        abstract for v1.0. 
    
    Returns
    -------
    str
    """
    filename = os.path.join(dataDir, "QuestionTypes", "{}_question_types.txt".
                            format(dataType))
    return filename

def img_file(dataSubType, imgId):
    """Image filename in the directory.
    
    Parameters
    ----------
    dataSubType : str
        "train2014" or "val2014" for example.
    imgId : int
        Image id.
    
    Returns
    -------
    str
    """
    filename = ".".join(("_".join(("COCO", dataSubType, str(imgId).zfill(12))), "jpg"))
    return filename

def glove_dir(dataDir):
    """Glove directory name.
    
    Parameters
    ----------
    dataDir : str
        Data directory path. 
    
    Returns
    -------
    str
    """
    dirname = os.path.join(dataDir, "Glove")
    return dirname

def vgg_embeddings_lstm_vgg_file(dataDir, taskType, dataType, dataSubTypes):
    """Embedded images file (.npy) path for the LSTM VGG inputs.
    
    Parameters
    ----------
    dataDir : str
        Data directory path.    
    taskType : str
        "OpenEnded" only for v2.0. "OpenEnded" or "MultipleChoice" for v1.0.
    dataType : str
        "mscoco" only for v1.0. "mscoco" for real and "abstract_v002" for 
        abstract for v1.0. 
    dataSubType : str, list
        As a list or a single string for one dataSubType ("train2014" or 
        "val2014" for example).
    
    Returns
    -------
    str
    """
    if type(dataSubTypes) == list:
        dataSubTypes = dataSubTypes
    else:
        dataSubTypes = [dataSubTypes]
     
    filename = os.path.join(dataDir, "Embeddings",
                            "{}_{}_{}_vgg_embeddings_lstm_vgg.npy"
                            .format(taskType, dataType, "_".join(dataSubTypes)))
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

def res_file(dataDir, taskType, dataType, dataSubType, methodName):
    """Results file path.
    
    Parameters
    ----------
    dataDir : str
        Data directory path.
    taskType : str
        "OpenEnded" only for v2.0. "OpenEnded" or "MultipleChoice" for v1.0.
    dataType : str
        "mscoco" only for v1.0. "mscoco" for real and "abstract_v002" for 
        abstract for v1.0. 
    dataSubType : str
        "train2014" or "val2014" for example.
    methodName : str
        Name of the Visual Question Answering method.
    
    Returns
    -------
    str
    """
    filename = os.path.join(dataDir, "Results", "{}_{}_{}_{}_results.json".
                                format(taskType, dataType, dataSubType, methodName))

    return filename

def save_results(results, dataDir, taskType, dataType, dataSubType, methodName):
    """Save the results obtained with the specific method in the Results 
    directory.
    
    Parameters
    ----------
    results : list (dict)
        List of dictionaries with the predicted answer for each question id :
        [{"answer": "no", "question_id": 1}].
    dataDir : str
        Data directory path.
    taskType : str
        "OpenEnded" only for v2.0. "OpenEnded" or "MultipleChoice" for v1.0.
    dataType : str
        "mscoco" only for v1.0. "mscoco" for real and "abstract_v002" for 
        abstract for v1.0. 
    dataSubType : str
        "train2014" or "val2014" for example.
    methodName : str
        Name of the Visual Question Answering method.
    
    Returns
    -------
    str
        Results file path.
    """
    print "--> Saving the results"

    results_file = res_file(dataDir, taskType, dataType, dataSubType, methodName)

    with open(results_file, "w") as f:
        json.dump(results, f)

    return results_file


