FROM python:3.7

WORKDIR /usr/app

COPY app.py utils.py requirements.txt templates static countries run_[]_cls_233_lr_0.001_bs_256_ts_8_tp_15288561_8620_model.hdf5 ./

RUN mkdir uploads

RUN ls

RUN pip install --upgrade pip

RUN pip install -r requirements.txt



EXPOSE 5000

CMD ["python", "app.py"]

