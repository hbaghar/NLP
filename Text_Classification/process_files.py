import os
from tqdm import tqdm
from text_processing import Document

def get_filepaths(dir):
    """
    Return file path for all txt files listed in a directory
    """
    return [os.path.join(dir,file) for file in os.listdir(dir) if file[-4:] == ".txt"]

dir = os.path.join(os.curdir, 'Text_Classification/review_polarity/txt_sentoken/pos')
for file in tqdm(get_filepaths(dir)):
    doc = Document(file)
    print(doc)
