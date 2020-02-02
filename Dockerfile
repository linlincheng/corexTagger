FROM python:3

RUN mkdir -p /opt/corex_tag

COPY requirement.txt /opt/corex_tag/requirement.txt
RUN pip install -r /opt/corex_tag/requirement.txt

COPY ./hastagger /opt/corex_tag/hashtagger/
COPY airline.csv /opt/corex_tag/airline.csv
COPY anchor_words.json /opt/corex_tag/anchor_words.json
COPY ./test /opt/corex_tag/test/
COPY ./new_playground.ipynb /opt/corex_tag/new_playground.ipynb
WORKDIR /opt/corex_tag
