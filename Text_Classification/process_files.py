import os
from tqdm import tqdm
from glob import glob
from text_processing import Document

def get_filepaths(root):
    """
    Return file path for all txt files listed in a directory and sub-directories
    """
    return [file for dir in os.walk(root) for file in glob(os.path.join(dir[0], '*.txt'))]

#Convert to input later
path = 'Text_Classification/review_polarity'

#We need to process all files and create a vocabulary
dir = os.path.join(os.curdir, path)
for file in tqdm(get_filepaths(dir)):
    pass

sentiment_map = {'pos':1, 'neg':-1}
