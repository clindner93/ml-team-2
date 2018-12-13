#This container contains your model and any helper scripts specific to your model.
FROM tensorflow/tensorflow:1.12.0

RUN pip install nltk
RUN pip install tflearn
RUN pip install flask

ADD Text_Classification.py /opt/Text_Classification.py
RUN chmod +x /opt/Text_Classification.py

ADD TrainingDataset.json /opt/TrainingDataset.json

ENTRYPOINT ["/usr/bin/python"]
CMD ["/opt/Text_Classification.py"]
