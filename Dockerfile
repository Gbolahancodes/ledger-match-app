FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Bootstrap: generate synthetic data and train the matcher at build time
# so the image is ready to serve immediately.
RUN python generate_data.py --scenario noisy --n 800 \
    && python -m src.train --scenario noisy --n 800

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
