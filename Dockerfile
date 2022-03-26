FROM python:3.10

RUN mkdir /dataset
COPY ./dataset /dataset

RUN mkdir /EXPERIMENTOS
WORKDIR /EXPERIMENTOS
COPY ./object_recognition /EXPERIMENTOS


RUN pip install numpy
RUN pip install opencv-python
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install scipy

CMD ["/EXPERIMENTOS/main.sh"]