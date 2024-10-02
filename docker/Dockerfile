FROM python:3.7-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git && apt-get clean

RUN git clone https://github.com/tsurumeso/vocal-remover .

RUN pip install -r requirements.txt