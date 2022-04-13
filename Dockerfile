FROM python:3.10

RUN mkdir /EXPERIMENTOS
WORKDIR /EXPERIMENTOS

RUN pip install numpy
RUN pip install opencv-python
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install scipy

RUN mkdir /dataset
CMD ["python", "/EXPERIMENTOS/main.py"]