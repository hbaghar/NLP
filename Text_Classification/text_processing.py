import os
from tqdm import tqdm
from time import sleep

def get_filepaths(dir):
    """
    Return file path for all txt files listed in a directory
    """
    return [os.path.join(dir,file) for file in os.listdir(dir) if file[-4:] == ".txt"]


class Document():
    """Contains word frequencies for an individual text document"""

    def __init__(self, filepath):
        self.__filepath = filepath
        self.__tokens = self.process_file()
        self.frequency_dict = self.tf()

    def __str__(self):
        end = self.__filepath.rfind('/')
        sent = None
        return "File name: " + self.__filepath[:]


    def process_file(self):
        """
        Perform text processing on a raw txt file given its filepath
        Returns a list of tokenized words
        """
        text = ''
        with open(self.__filepath) as f:
            text = f.readlines()
        text = ' '.join(text)
        return text.split(' ')

    def word_count(self, token):
        return self.__tokens.count(token)

    def tf(self):
        word_dict = {}
        for token in self.__tokens:
            word_dict[token] = self.word_count(token)/len(self.__tokens)
        return word_dict

dir = os.path.join(os.curdir, 'Text_Classification/review_polarity/txt_sentoken/pos')
for file in tqdm(get_filepaths(dir)[:5]):
    doc = Document(file)
    print(doc.frequency_dict)
