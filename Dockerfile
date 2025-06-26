# Dockerfile

# --- Stage 1: Base Image ---
# Use an official Python slim image as a parent image. 'slim' versions are smaller
# as they don't include common packages you might not need.
FROM python:3.9-slim

# --- Stage 2: Set up the working environment ---
# Set the working directory inside the container. All subsequent commands will
# be run from this directory.
WORKDIR /app

# --- Stage 3: Install Dependencies ---
# Copy only the requirements files first. This is a Docker caching optimization.
# If these files don't change, Docker can reuse the cached layer from a previous build,
# making subsequent builds much faster.
COPY requirements.txt requirements.txt
# (We don't need requirements-dev.txt in the final image, as it's not needed to run the toolkit)

# Run pip install to get the runtime dependencies.
# --no-cache-dir: Reduces image size by not storing the pip cache.
# --upgrade pip: Good practice to ensure pip is up-to-date.
RUN python -m pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Stage 4: Copy Application Code ---
# Now that dependencies are installed, copy the application source code into the container.
# The dot '.' at the end means "copy everything from the 'pyreliabilitypro' directory
# in the build context to the 'pyreliabilitypro' directory inside the container's /app directory".
COPY pyreliabilitypro/ ./pyreliabilitypro

# --- Stage 5: (Optional) Set up an Entrypoint for a CLI ---
# This defines the default command to run when the container starts.
# We'll set it up as if we have a CLI, which is good practice.
# If you don't have a CLI, the container can still be used for running Python scripts.
# For now, we'll make the default command an interactive Python shell.
CMD ["python"]