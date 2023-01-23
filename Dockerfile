# 
FROM python:3.10 as base
# 
# Setup env
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONFAULTHANDLER 1
# Install & use pipenv
COPY Pipfile Pipfile.lock ./
#zipp y r u like this?
RUN python -m pip install --upgrade pip
RUN pip install pipenv && pipenv install --dev --system --deploy && pipenv lock --clear
# Create and switch to a new user
RUN useradd --create-home appuser
WORKDIR /home/appuser
USER appuser
# Copy app files and the actual application
COPY ./src ./src 
COPY ./models ./models
COPY ./data ./data
#
CMD ["pipenv", "run", "python", "/src/bot.py"]