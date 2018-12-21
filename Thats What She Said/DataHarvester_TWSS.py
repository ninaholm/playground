
from bs4 import BeautifulSoup
import requests
import pickle


baseURL = "http://www.twssstories.com"
linklist = ["/"]
twss = []

while linklist:
    
    url = baseURL + linklist.pop()
    
    sourceCode = requests.get(url)
    soup = BeautifulSoup(sourceCode.text, "lxml")
    
    
    for link in soup.findAll('a', {'title': 'Go to next page'}):
        linklist.append(link.get('href'))
        
    for snippet in soup.findAll('div', { 'class': 'content clear-block'}):
        text = snippet.p.text
        text = text.replace('TWSS', '')
        twss.append(text)
    

with open("C:/Users/NHJ/Desktop/Playground/app/modules/TWSSlist.dat", 'wb') as f:
        pickle.dump(twss, f)

with open("C:/Users/NHJ/Desktop/Playground/app/modules/TWSSlistDoubleCheck.txt", 'w', encoding='UTF-8') as f:
        f.write("\n".join(twss))






