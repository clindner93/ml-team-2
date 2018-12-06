#This container contains your model and any helper scripts specific to your model.
FROM tensorflow/tensorflow:1.7.0

RUN pip install pandas
RUN pip install numpy
RUN pip install sklearn

ADD train.py /opt/train.py
RUN chmod +x /opt/train.py

ENTRYPOINT ["/usr/bin/python"]
CMD ["/opt/train.py"]
