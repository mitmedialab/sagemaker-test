FROM python:3.8

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ .

RUN cp -r /src /opt/program/
COPY src/serve.py /opt/program/serve
RUN chmod +x /opt/program/serve

WORKDIR /opt/program
