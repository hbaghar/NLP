import os
from tqdm import tqdm
from glob import glob
from text_processing import Document
import numpy as np

class ProcessDocuments(object):
    """docstring for ProcessDocuments."""
    #Remove root and pass location as parameter for each funtion
    def __init__(self, train_path = 'Text_Classification/data/train', test_path = 'Text_Classification/data/test'):
        self.train_path = train_path
        self.test_path = test_path
        self.__train_documents = []
        self.__test_documents = []
        self.__word_doc_freq = {}
        self.vocab = None
        self.train_data = None
        self.test_data = None

    def __get_filepaths(self, path):
        """
        Return file path for all txt files listed in a directory and sub-directories
        """
        return [file for dir in os.walk(path) for file in glob(os.path.join(dir[0], '*.txt'))]
    #Return  value and call separately for train and test data
    def __get_documents(self, path):
        """
        Create a list of document objects for each text file
        """
        return [Document(file) for file in tqdm(self.__get_filepaths(path), desc = "Processing documents")]

    #Call when training data is being created
    def __get_vocab(self):
        """
        Function that will gather all the words from each document and build a bag of words
        """
        for doc in tqdm(self.__train_documents, desc = "Generating vocabulary"):
            for token in doc.tf().keys():
                self.__word_doc_freq[token] = self.__word_doc_freq.get(token, 0) + 1

        #Creating lookup table using dictionary for faster search for idx in get_dataset
        self.vocab = {token: i for token,i in zip(self.__word_doc_freq.keys(), range(len(self.__word_doc_freq)))}

    def __get_data_matrix(self, docs):
        """
        Creates dataset in the form of a numpy array. Features are tf-idf scores and last column is sentiment label
        """
        #Define matrix size based on document and feature counts
        x = np.zeros((len(docs), len(self.__word_doc_freq)))
        y = np.zeros((len(docs), 1))
        #Split into function and allow passing custom list of document objects (so that it can be re-run for train and test)
        #Iterating every document
        for i,doc in enumerate(tqdm(docs, desc = "Creating dataset")):
            #Iterating every word in document
            for token, val in doc.tf().items():
                try:
                    idx = self.vocab[token]
                    #Calculating tf-idf score
                    x[i,idx] = val*np.log(len(docs)/(self.__word_doc_freq[token]+1))
                except:
                    pass
                #Assigning labels
                y[i] = 1 if doc.sentiment == "pos" else -1

        return x,y

    def get_train_data(self):
        print("Processing training data files...")
        self.__train_documents = self.__get_documents(self.train_path)
        self.__get_vocab()

        print("\nCreating training dataset...")
        self.train_data = self.__get_data_matrix(self.__train_documents)

        return self.train_data

    def get_test_data(self):
        print("\nProcessing test data files...")
        self.__test_documents = self.__get_documents(self.test_path)

        print("\nCreating test dataset...")
        self.test_data = self.__get_data_matrix(self.__test_documents)

        return self.test_data
