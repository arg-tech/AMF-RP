FROM python:3

RUN mkdir -p /home/AMF-RP
WORKDIR /home/AMF-RP

RUN pip install --upgrade pip


ADD requirements.txt .
RUN pip install -r requirements.txt
RUN pip install gunicorn

ADD app app
ADD app/svm_pipeline.joblib app/svm_pipeline.joblib

ADD boot.sh ./
RUN chmod +x boot.sh


ENV FLASK_APP app


EXPOSE 5000
ENTRYPOINT ["./boot.sh"]