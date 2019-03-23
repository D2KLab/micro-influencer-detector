from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import os

def parseFastText(path):
    #testData = []
    #testWords = []
    data = {}
    dataset = open(path, "r")
    [num_words, embedding_size] = dataset.readline().split(" ")
    i=1
    print("Reading dataset file...")
    for line in dataset:
        [word, embedding] = line[:-1].split(" ",1)
        data[word] = embedding
        percentage = round( (100*(i/int(num_words))),1 )
        i += 1
        # print("  %.1f%% complete" %percentage)

    print("\n")
    dataset.close()
    return data
    #return [np.array(testData), testWords]

def loadEmbeddingsDataset(path, binaryFormat):
    if binaryFormat:
        return KeyedVectors.load_word2vec_format(path, binary=True)
    else:
        return KeyedVectors.load_word2vec_format(path)

def readMyPersonality():
    data = []
    for line in open("dataset/statuses_unicode.txt", "r"):
        data.append(line[:-1])
    data = np.array(data)

    y_O = np.zeros(1)
    y_C = np.zeros(1)
    y_E = np.zeros(1)
    y_A = np.zeros(1)
    y_N = np.zeros(1)
    i=0
    for line in open("dataset/big5labels.txt", "r"):
        big5_str = line[:-1].split(" ")
        if i==0:
            y_O[0] = float(big5_str[0])
            y_C[0] = float(big5_str[1])
            y_E[0] = float(big5_str[2])
            y_A[0] = float(big5_str[3])
            y_N[0] = float(big5_str[4])
        else:
            y_O = np.append(y_O, float(big5_str[0]))
            y_C = np.append(y_C, float(big5_str[1]))
            y_E = np.append(y_E, float(big5_str[2]))
            y_A = np.append(y_A, float(big5_str[3]))
            y_N = np.append(y_N, float(big5_str[4]))
        i += 1

    return [data, y_O, y_C, y_E, y_A, y_N]

def readMyPersonalityUserWise():
    #load data for test 3-4 (unique string for all user statuses)
    data_all = []
    y_O_all = np.zeros(1)
    y_C_all = np.zeros(1)
    y_E_all = np.zeros(1)
    y_A_all = np.zeros(1)
    y_N_all = np.zeros(1)
    old_id = ""
    text = ""
    for line in open("dataset/mypersonality_final.csv", "r"):
        user_id = line.split(',')[0][1:]
        status = line.split(user_id)[1][3:].split('"",')[0]
        big5_str = line.split(status)[1][3:].split(',""')[0].split(",")
        if old_id == "":
            old_id = user_id
            text = text + " " + status
            y_O_all = float(big5_str[0])
            y_C_all = float(big5_str[1])
            y_E_all = float(big5_str[2])
            y_A_all = float(big5_str[3])
            y_N_all = float(big5_str[4])
        elif old_id != user_id:
            data_all.append(text)
            y_O_all = np.append(y_O_all, float(big5_str[0]))
            y_C_all = np.append(y_C_all, float(big5_str[1]))
            y_E_all = np.append(y_E_all, float(big5_str[2]))
            y_A_all = np.append(y_A_all, float(big5_str[3]))
            y_N_all = np.append(y_N_all, float(big5_str[4]))
            text = status
            old_id = user_id
        else:
            text = text + " " + status
    data_all.append(text)

    return [data_all, y_O_all, y_C_all, y_E_all, y_A_all, y_N_all]

