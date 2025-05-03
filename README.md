# News-vs-Troll-Content_Classifier

Set of models to test various neural archietcture for classifying between factual news and troll comments.

### SETUP

Initialize conda enviroment with
```
ml miniconda
conda env create -f environment.yml
conda activate news-vs-troll
```

If certain dependencies fail to correctly install (an issuie I was using while on Grace), use the pip install to directly install them to your enviroment. If you wish to view specific dependencies and versions, they can be viewed in the enviorment.yml file.

```
For Example:
pip install --no-cache-dir \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  torch==2.6.0+cu118 \
  transformers==4.48.2 \
  datasets==3.2.0 \
```

Download and unzip GloVe embeddings.
```
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip glove.6B.100d.txt
```

Finally, run the script.
```
python news-versus-troll.py
```
