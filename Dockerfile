FROM continuumio/anaconda:latest

RUN apt-get -qq install libc-dev lsb-release && \
    conda install seaborn pyside --yes
RUN apt-get -qq install g++ gcc make && \
    conda install gcc && \
    pip install xgboost
RUN conda install -c conda-forge tensorflow scikit-learn=0.18 'icu=58.*' lxml geopandas --yes
RUN pip install keras
RUN conda install -c bokeh bokeh=0.12.6

ADD jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py
