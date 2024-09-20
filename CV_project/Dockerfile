# app/Dockerfile

# Use the official Python image with version 3.9
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the local code into the container
COPY . .

# Install the Python dependencies from the requirements.txt file
RUN pip3 install --no-cache-dir -r requirements.txt

# Expose the Streamlit default port
EXPOSE 8501

# Define a health check to ensure the app is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Set the entry point to run the Streamlit app on container startup
ENTRYPOINT ["streamlit", "run", "cv_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
