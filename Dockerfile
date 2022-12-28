# 
FROM python:3.10
# 
RUN pip install pipenv
#
COPY Pipfile .
COPY Pipfile.lock .
RUN PIPENV_VENV_IN_PROJECT=1 pipenv install --deploy
#
FROM superfan-bot:latest AS runtime
# Copy virtual env from python-deps stage
COPY --from=superfan-bot /.venv /.venv
ENV PATH="/.venv/bin:$PATH"
# Create and switch to a new user
RUN useradd --create-home appuser
WORKDIR /home/appuser
USER appuser
# 
COPY ./app /superfan-bot/app
# 
COPY ./models /superfan-bot/models
#
COPY ./data /superfan-bot/data
#
ENTRYPOINT ["python", "-m", "http.server"]
CMD ["pipenv", "shell", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]