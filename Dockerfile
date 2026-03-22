FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY proxy.py .
COPY auth.py .
COPY audit.py .
COPY plugins/ ./plugins/

EXPOSE 8080
CMD ["python", "proxy.py"]
