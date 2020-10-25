FROM tensorflow/tensorflow:latest

WORKDIR /usr/app

COPY requirements.txt ./

RUN pip install -r requirements.txt

RUN yes | apt update && yes | apt install libgl1-mesa-glx

RUN mkdir uploads

COPY templates templates
COPY static static
COPY countries countries

COPY app.py utils.py run_[[]]_cls_233_lr_0.001_bs_256_ts_8_tp_15288561_8620_model.hdf5 ./

RUN cat /etc/os-release

EXPOSE 5000

COPY countries/data static/data

RUN find .

CMD exec gunicorn --bind :5000 --workers 1 --threads 8 app:app

