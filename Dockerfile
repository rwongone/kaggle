FROM continuumio/anaconda:latest

RUN apt-get -qq install libc-dev lsb-release && \
    conda install seaborn pyside --yes
RUN apt-get -qq install g++ gcc make && \
    conda install gcc && \
    pip install xgboost
