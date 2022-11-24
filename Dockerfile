FROM python:3.10-slim-bullseye

RUN pip3 install --upgrade pip

WORKDIR /app

COPY . /app

RUN pip3 --no-cache-dir install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow_cpu-2.11.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

RUN pip3 --no-cache-dir install -r requirements.txt

EXPOSE 5000

CMD ["python3", "app.py"]
