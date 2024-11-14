FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime
WORKDIR /app

COPY requirements.txt /app/

RUN pip install -r requirements.txt


COPY mlops.py /app/

ENTRYPOINT ["python", "/app/mlops.py"]

CMD ["python", "mlops.py"]
