import os
import re

class Document():
    """Contains word frequencies for an individual text document"""

    def __init__(self, filepath):
        self.__filepath = filepath
        self.sentiment = filepath[:filepath.rfind('/')][-3:]
        self.tokens = self.__process_file()
        self.num_words = self.get_num_words()

    def __str__(self):
        end = self.__filepath.rfind('/')
        return f"File name: %s (%s) (%d words)" % (self.__filepath[end+1:], self.sentiment, self.num_words)

    def __process_file(self):
        """
        Perform text processing on a raw txt file given its filepath
        Returns a list of tokenized words after removing special characters
        """
        text = ''
        with open(self.__filepath) as f:
            text = f.readlines()
        text = ' '.join(text)
        text = re.sub('[^a-zA-Z\d\s]+', "", text)
        return [token for token in text.split(' ') if token != '']

    def __word_count(self, token):
        """
        Counts occurences of a specific word
        """
        return self.tokens.count(token)

    def get_num_words(self):
        """
        Returns number of words in document
        """
        return len(self.tokens)

    def tf(self):
        """
        Returns a dict containing term frequency normalized by total number of words in document
        """
        word_dict = {}
        for token in set(self.tokens):
            word_dict[token] = self.__word_count(token)/self.num_words
        return word_dict
