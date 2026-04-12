## Stage 1: Build React frontend
FROM node:20-slim AS frontend
WORKDIR /build
COPY webapp/frontend/package.json webapp/frontend/package-lock.json ./
RUN npm ci
COPY webapp/frontend/ ./
RUN npm run build

## Stage 2: Python backend + serve static frontend
FROM python:3.12-slim
WORKDIR /app

COPY webapp/backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY webapp/backend/ ./

# Copy ML project modules that backend imports via sys.path
COPY envs/ /project/envs/
COPY agent/ /project/agent/
COPY src/ /project/src/

# Copy built frontend into backend/static
COPY --from=frontend /build/dist ./static

EXPOSE 8000

CMD ["python", "app.py"]
