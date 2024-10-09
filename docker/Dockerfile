FROM nvcr.io/nvidia/pytorch:24.09-py3

ENV https_proxy "http://http-p.srv.cc.suzuka-ct.ac.jp:8080"
ENV http_proxy "http://http-p.srv.cc.suzuka-ct.ac.jp:8080"

RUN apt-get update && apt-get install -y git && apt-get clean

RUN git clone https://github.com/tsurumeso/vocal-remover

RUN pip install -r vocal-remover/requirements.txt