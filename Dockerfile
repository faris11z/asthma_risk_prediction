FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY . .

EXPOSE 7860

CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:7860", "app:app"]
