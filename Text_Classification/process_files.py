import os
from tqdm import tqdm
from glob import glob
from text_processing import Document
import numpy as np

class ProcessDocuments(object):
    """docstring for ProcessDocuments."""

    def __init__(self, root='Text_Classification/review_polarity'):
        self.__root = root
        self.__documents = []
        self.__word_doc_freq = {}
        self.vocab = None
        self.data = None

    def __get_filepaths(self):
        """
        Return file path for all txt files listed in a directory and sub-directories
        """
        return [file for dir in os.walk(self.__root) for file in glob(os.path.join(dir[0], '*.txt'))]

    def __get_documents(self):
        """
        Create a list of document objects for each text file
        """
        self.__documents = [Document(file) for file in tqdm(self.__get_filepaths(), desc = "Processing documents")]

    def __get_vocab(self):
        """
        Function that will gather all the words from each document and build a bag of words
        """
        for doc in tqdm(self.__documents, desc = "Generating vocabulary"):
            for token in doc.tf().keys():
                self.__word_doc_freq[token] = self.__word_doc_freq.get(token, 0) + 1

        self.vocab = {token: i for token,i in zip(self.__word_doc_freq.keys(), range(len(self.__word_doc_freq)))}

    def get_dataset(self):
        self.__get_documents()
        self.__get_vocab()

        self.data = np.zeros((len(self.__documents), len(self.__word_doc_freq)+1))
        for i,doc in enumerate(tqdm(self.__documents, desc = "Craeating dataset")):
            for token, val in doc.tf().items():
                idx = self.vocab[token]
                self.data[i,idx] = val*np.log(len(self.__documents)/(self.__word_doc_freq[token]+1))
                self.data[i,-1] = 1 if doc.sentiment == "pos" else -1

def main():
    processor = ProcessDocuments()
    processor.get_dataset()

if __name__ == "__main__":
    main()
