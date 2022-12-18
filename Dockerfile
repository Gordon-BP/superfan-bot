# 
FROM python:3.10
# 
WORKDIR /superfan-bot
# 
COPY ./requirements.txt /superfan-bot/requirements.txt
# 
RUN pip install --no-cache-dir --upgrade -r /superfan-bot/requirements.txt
# 
COPY ./app /superfan-bot/app
# 
COPY ./data /superfan-bot/data
# 
COPY ./models /superfan-bot/models
#
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]