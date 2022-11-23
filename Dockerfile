FROM python:3.9-bullseye

RUN python3 -m ensurepip \
    && pip3 install --upgrade pip

WORKDIR /app

COPY . /app

RUN pip3 --no-cache-dir install -r requirements.txt

EXPOSE 5000

CMD ["python3", "app.py"]
