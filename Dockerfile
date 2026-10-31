FROM python:3.11-slim AS runtime
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml README.md ./
COPY src ./src
COPY app.py ./
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .
EXPOSE 7860
CMD ["python", "app.py"]
