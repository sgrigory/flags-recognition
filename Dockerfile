FROM tensorflow/tensorflow:latest

WORKDIR /usr/app

COPY requirements.txt ./

RUN pip install -r requirements.txt

RUN yes | apt update && yes | apt install libgl1-mesa-glx

RUN mkdir uploads

COPY templates templates
COPY static static
COPY app_data app_data

COPY app.py utils.py ./

RUN cat /etc/os-release

EXPOSE 5000

COPY app_data/PNG-128 static/app_data/PNG-128

RUN find .

CMD exec gunicorn --bind :5000 --workers 1 --threads 8 app:app

