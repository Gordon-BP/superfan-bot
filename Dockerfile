# 
FROM python:3.10-slim as base
# 
# Setup env
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONFAULTHANDLER 1
# Install & use pipenv. No Pipfile.lock because I was getting SHA failures
COPY Pipfile Pipfile.lock ./
RUN apt update
RUN apt install -y p7zip-full 
RUN python -m pip install --upgrade pip --no-cache-dir
RUN pip install pipenv 
RUN PIPENV_VENV_IN_PROJECT=1 pipenv install --deploy && pipenv lock --clear
# Copy app files and the actual application
COPY ./src /src 
COPY ./models /models 
# Create user and own files
USER root
RUN adduser appuser
RUN mkdir data && chown appuser data
USER appuser
# Define ENV variables
ENV DISCORD_TOKEN = ""
ENV COHERE_API_KEY = ""
ENV PINECONE_API_KEY = ""
ENV PINECONE_ENV = "us-west1-gcp"
ENV PINECONE_NAME = "default"
# Start the bot
CMD ["pipenv", "run", "python", "/src/bot.py"]