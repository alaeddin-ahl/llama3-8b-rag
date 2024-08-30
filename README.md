## Download ollama
https://ollama.com/

## Install and run llama3 8b instruct in terminal
```
ollama run llama3.1
```

## 1. Setup virtual environment for python
```
virtualenv env
source ./env/bin/activate
pip install --upgrade pip
```

## 2. Install ollama and langchain_community package in the virtual environment
```
pip install -r requirements.txt
```
(ollama lets us easily use llama3 8b, langchain_community offers sufisticated functions to load, split and embed pdfs)


## 3. Put PDFs in pdfs/ sub-folder + adjust prompt

## 4. run  python script
```
python local_rag.py
```
