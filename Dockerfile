FROM python:3.10-slim

WORKDIR /code

# Copy and install requirements
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy all files to the container
COPY . .

# Run Streamlit on port 7860 (required by Hugging Face Spaces)
CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
