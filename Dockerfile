FROM python:3.8-slim-buster
ENV FLASK_APP app
ENV FLASK_ENV production
WORKDIR /app
COPY . .
COPY corpora /root/nltk_data/corpora
RUN pip3 install -r requirements.txt
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
