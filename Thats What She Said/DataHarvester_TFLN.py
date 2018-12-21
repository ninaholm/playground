
from bs4 import BeautifulSoup
import requests
import pickle


baseURL = "http://www.textsfromlastnight.com/texts/page:"
tfln = []
i = 1

while i < 3000:
    
    if i % 100 == 0:
        print(i)

    url = baseURL + str(i)
    i = i+1
    
    print(url)
    
    sourceCode = requests.get(url)
    soup = BeautifulSoup(sourceCode.text, "lxml")
        
    for snippet in soup.findAll('div', { 'class': 'text'}):
        text = snippet.p.a
        
        if text is not None:
            tfln.append(str(text.string))


with open("C:/Users/NHJ/Desktop/Playground/app/modules/TFLNlist.dat", 'wb') as f:
        pickle.dump(tfln, f)

with open("C:/Users/NHJ/Desktop/Playground/app/modules/TFLNlistDoubleCheck.txt", 'w', encoding='UTF-8') as f:
        f.write("\n".join(tfln))
        





