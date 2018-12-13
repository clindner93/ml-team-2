#This container contains your model and any helper scripts specific to your model.
FROM tensorflow/tensorflow:1.12.0

RUN pip install -U nltk
RUN pip install tflearn
RUN pip install flask

ADD init.py /opt/init.py
RUN python /opt/init.py

ADD Text_Classification.py /opt/Text_Classification.py
RUN chmod +x /opt/Text_Classification.py

ENTRYPOINT ["/usr/bin/python"]
CMD ["/opt/Text_Classification.py"]
