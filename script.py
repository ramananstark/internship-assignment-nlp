# ## Import all the dependent libraries
# import os
# import tensorflow as tf
# import 

import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer


# def get_mca_questions(context: str):
#    mca_questions = ["mca1","mca2","mca3"]p
   
#    return mca_questions  

import fitz

pdffile=fitz.open("C:\\Users\\Sivaprasath\\Desktop\\RAMANAN\\akaike\\internship-assignment-nlp\\Dataset\\chapter-2.pdf")

extracted=''

for page_num in range(pdffile.page_count):
    page=pdffile.load_page(page_num)
    text=page.get_text()
    extracted+=text

print(extracted)

tokenizer=Tokenizer(num_words=10000,oov_token=<OOV>)