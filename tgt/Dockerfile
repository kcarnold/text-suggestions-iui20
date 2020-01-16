FROM python:3.6-buster AS iui20-basedeps

RUN pip install six tqdm future
RUN pip install torch==0.3.1 nltk==3.2.5 torchtext==0.2.3 tornado==5.1 h5py==2.9
RUN mkdir vendor && cd vendor && wget -q https://github.com/kcarnold/OpenNMT-py/archive/vecs.zip && unzip -q vecs.zip && mv OpenNMT-py-vecs OpenNMT-py && rm vecs.zip
RUN curl -L https://github.com/explosion/spacy-models/releases/download/en_core_web_md-2.1.0/en_core_web_md-2.1.0.tar.gz | (cd vendor && tar zxv) 
RUN pip install vendor/OpenNMT-py vendor/en_core_web_md-2.1.0
RUN mkdir logs
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('perluniprops'); nltk.download('nonbreaking_prefixes')"
COPY . .
RUN pip install -e .
