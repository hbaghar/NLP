import os
from tqdm import tqdm
from glob import glob
from text_processing import Document

class ProcessDocuments(object):
    """docstring for ProcessDocuments."""

    def __init__(self, root='Text_Classification/review_polarity'):
        self.__root = root
        self.__documents = []
        self.vocab = {}

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
        for doc in tqdm(self.__documents):


    def run(self):
        self.__get_documents()
        self.__get_vocab()


def main():
    processor = ProcessDocuments()
    processor.run()

if __name__ == "__main__":
    main()
