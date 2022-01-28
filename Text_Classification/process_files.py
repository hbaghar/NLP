import os
from tqdm import tqdm
from glob import glob
from text_processing import Document
import numpy as np

class ProcessFiles(object):
    """
    Creates training and test data for a folder containing raw txt files.

    Text files should be housed in a folder structure as follows:

    .
    |-data
    |    |_ train
    |    |   |_ pos
    |    |   |_ neg
    |    |_ test
    |        |_ pos
    |        |_ neg
    |_ process_files.py
    |_ text_processing.py
    """
# TODO: Add bias term in dataset
# TODO: Limit vocabulary to top n features
    def __init__(self, train_path = 'Text_Classification/data/train', test_path = 'Text_Classification/data/test'):
        self.train_path = train_path
        self.test_path = test_path
        self.__train_files = []
        self.__test_files = []
        self.__word_doc_freq = {}
        self.vocab = None
        self.train_data = None
        self.test_data = None

    def __get_filepaths(self, path):
        """
        Return file path for all txt files listed in a directory and sub-directories
        """
        return [file for dir in os.walk(path) for file in glob(os.path.join(dir[0], '*.txt'))]

    def __get_vocab(self):
        """
        Function that will gather all the words from each document and build a bag of words
        """
        for file in tqdm(self.__train_files, desc = "Generating vocabulary"):
            doc = Document(file)
            for token in doc.tf().keys():
                self.__word_doc_freq[token] = self.__word_doc_freq.get(token, 0) + 1

        #Creating lookup table using dictionary for faster search for idx in get_dataset
        self.vocab = {token: i for token,i in zip(self.__word_doc_freq.keys(), range(len(self.__word_doc_freq)))}

    def __get_data_matrix(self, files):
        """
        Creates dataset in the form of a numpy array. Features are tf-idf scores and last column is sentiment label
        """
        #Define matrix size based on document and feature counts
        x = np.zeros((len(files), len(self.__word_doc_freq)))
        y = np.zeros((len(files), ))
        #Iterating over every document
        for i, file in enumerate(tqdm(files, desc = "Creating dataset")):
            #Iterating every word in document
            doc = Document(file)
            for token, val in doc.tf().items():
                try:
                    idx = self.vocab[token]
                    #Calculating tf-idf score
                    x[i,idx] = val#*np.log(len(files)/(self.__word_doc_freq[token]+1))
                except:
                    pass
                #Assigning labels
                y[i] = 1 if doc.sentiment == "pos" else 0

        return x,y

    def get_train_data(self):
        print("Processing training data files...")
        self.__train_files = self.__get_filepaths(self.train_path)
        self.__get_vocab()

        print("\nCreating training dataset...")
        self.train_data = self.__get_data_matrix(self.__train_files)

        return self.train_data

    def get_test_data(self):
        print("\nProcessing test data files...")
        self.__test_files = self.__get_filepaths(self.test_path)

        print("Creating test dataset...")
        self.test_data = self.__get_data_matrix(self.__test_files)

        return self.test_data
