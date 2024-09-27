# Use the official Python image as a base
FROM python:3.7

WORKDIR /app/code

COPY requirements.txt /app/code/

RUN pip install --no-cache-dir -r /app/code/requirements.txt

COPY code /app/code/

CMD ["python", "main_index.py"]
