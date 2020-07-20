FROM pytorch/pytorch

##Set environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN pip install requests
RUN pip install numpy
RUN pip install flask
RUN pip install flask-cors
RUN pip install textstat
RUN pip install dynaconf
RUN pip install regex
RUN pip install six
RUN pip install lxml
RUN pip install cchardet
RUN pip install chardet
RUN pip install html5lib
RUN pip install cssselect
#### Make port 5000 available to the world outside this container
EXPOSE 5000
#### create a folder where files will be stored
#RUN mkdir /opt/
RUN mkdir -p /opt/ReplyNNService
#### copying files
COPY . /opt/ReplyNNService
#### run file
WORKDIR /opt/ReplyNNService
CMD ["python", "/opt/ReplyNNService/run.py"]
####
