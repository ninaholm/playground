

import spacy

from spacy.lang.da import Danish
from spacy.lang.en import English

import os

os.chdir("C:/Users/NHJ/Desktop/playground/DanishPreProcessing/")





nlp_en = English()


nlp_da = Danish()

doc = nlp_da(u'Fisk er en god spise, især på en torsdag med kage.')

for token in doc:
    print(f"{token.text} | {token.lemma_} | {token.pos_}")


# Get POS data

import xml.etree.ElementTree as etree
from bs4 import BeautifulSoup


TRAIN_DATA = [
    ('Who is Shaka Khan?', {
        'entities': [(7, 17, 'PERSON')]
    }),
    ('I like London and Berlin.', {
        'entities': [(7, 13, 'LOC'), (18, 24, 'LOC')]
    })
]

ddt_data_filename = 'data/ddt-1.0.xml'
    
with open(ddt_data_filename, "r") as file:
    content = file.read() # xml content stored in this variable and decode to utf-8

soup = BeautifulSoup(content, 'html.parser') #parse content to BeautifulSoup Module
terminallist = [terminals.findAll("t") for terminals in soup.findAll("terminals")]
data = [tpl for sublist in terminallist for tpl in sublist]

data = [(x["lemma"], x["cat"]) for x in data]




[item for sublist in l for item in sublist]



    


# POS-tagging

optimizer = nlp.begin_training(get_data)
for itn in range(100):
    random.shuffle(train_data)
    for raw_text, entity_offsets in train_data:
        doc = nlp.make_doc(raw_text)
        gold = GoldParse(doc, entities=entity_offsets)
        nlp.update([doc], [gold], drop=0.5, sgd=optimizer)
nlp.to_disk('/model')



