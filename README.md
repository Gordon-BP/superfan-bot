# superfan-bot
A conversational AI that reads a fandom's wiki and becomes an expert for your Discord server!

Big shoutouts to OpenAI, Kern AI, and stack overflow for their amazing sample code:
* [Question Answering using Embeddings](https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb)
* [Collect Wikipedia data about Olympic Games 2020](https://github.com/openai/openai-cookbook/blob/838f000935d9df03e75e181cbcea2e306850794b/examples/fine-tuned_qa/olympics-1-collect-data.ipynb)
* [Fine-tuning embeddings for better similarity search](https://github.com/code-kern-ai/refinery-sample-projects/tree/finetuning-similarity-search)


## Here's what actually works:
1. A cool `post` endpoint that can get answers to questions about the Civilisation video game franchise!
2. A script for parsing an XML data export from a fandom wiki (and cleaning it up a little bit)
    > This script can also query OpenAI's new embeddings endpoint to get embeddings for the article sections! So far I've never spend more than 0.10 USD embedding the data

## What's left?
* Add a column to the articles database with the article URL so we can link to it in the answer.
* Mount the embeddings and articles dataframes in some kind of database that doesn't have to reload every time I restart the app
* Dockerize the app and the database so they can run on a server that's 1000X more powerful than my poor macbook air
* Set up a discord bot that regurgitates GPT's answer along with some fancy formatting in a discord server.
* Optimize the cleanin regex patterns to get rid of all the moustache placeholders and other formatting BS

## How to get it all set up:

1. Download the wiki XML. The program is set for fandom wiki exports from whatever-place.fandom.com/special:statistics. They come as a 7zip file and you will need to unpack it on your own.
2. Run the 'createData.py' script on your data. I think I made an endpoint for it in the FastAPI?
3. Use the embeddings endpoint to generate embeddings for your article database. This takes 30 - 60 min depending on the dataset size.
4. Mount your embeddings and articles databases somewhere fast and accessible (I haven't done this yet and its killing meeeee)
5. Start FastAPI and go to the endpoint
6. `post` your question and wait for the results!