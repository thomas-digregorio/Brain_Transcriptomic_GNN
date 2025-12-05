# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git build-essential

# Install python dependencies
# We copy requirements first to leverage Docker cache
COPY environment.yml .
# Note: Installing via conda in docker can be slow/large. 
# Ideally we export to requirements.txt, but here we use conda for consistency.
RUN conda env create -f environment.yml

# Activate environment
SHELL ["conda", "run", "-n", "brain-gnn", "/bin/bash", "-c"]

# Copy the rest of the application
COPY . .

# Expose port for API
EXPOSE 8000

# Command to run the API
CMD ["uvicorn", "deploy.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
