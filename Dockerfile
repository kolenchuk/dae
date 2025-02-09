FROM python:3.9-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

# Install packages as root first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create and switch to non-root user after installing packages
RUN adduser --disabled-password appuser
COPY . .

# Fix permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

EXPOSE 8000

# Use the full path to uvicorn
CMD ["sh", "-c", "python ./src/bart_loader.py && /usr/local/bin/uvicorn app:app --host 0.0.0.0 --port 8000"]