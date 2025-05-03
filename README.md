# News-vs-Troll-Content_Classifier

Set of models to test various neural archietcture for classifying between factual news and troll comments

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

Ensure that the file containg GloVe embeddings is unzipped.
`unzip glove.6B.zip`

Finally, run the script.
`python news-versus-troll.py`
