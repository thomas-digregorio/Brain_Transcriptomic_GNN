# Usage:
# From the root of the repository:
# docker build -f deploy/Dockerfile -t brain-gnn-app .
# docker run -p 8000:8000 brain-gnn-app

# Use an official PyTorch runtime as a parent image
# Update to CUDA 12.1 to match project requirements
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git build-essential

# Install python dependencies
# We copy environment.yml from root
COPY environment.yml .
RUN conda env create -f environment.yml

# Check if we need to modify the SHELL to activate the env for subsequent commands
# (The previous SHELL instruction does this for RUN commands)
SHELL ["conda", "run", "-n", "brain-gnn", "/bin/bash", "-c"]

# Copy everything from root (build context should be root)
COPY . .

# Expose port for API
EXPOSE 8000

# Command to run the API
# explicit path to python in the env or use conda run
CMD ["conda", "run", "--no-capture-output", "-n", "brain-gnn", "uvicorn", "deploy.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
