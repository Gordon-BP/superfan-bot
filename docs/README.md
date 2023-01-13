#Hello World!

##This is a barebones static webpage for the project

I hope to include a bit more documentation here (eventually) and maybe an interactive widget if I can swing it.

Until then, I'm just going to copy the content from the main readme file....

# superfan-bot
Turn your favorite fandom wiki into a conversational AI!
Though it doesn't _have_ to be a fandom wiki, it could be any XML file. Or HTML. Or Markdown. Just gotta write a parsing program for those....

This project is built on top of Google Cloud and will require a GCP account with billing set up to use.

## This project built to be production-ready!
Jupyter Notebooks are for kindergardeners, this project is built for the real-world with production-ready features like:
1. Built on Google Cloud Project, so it can scale with the power of the Goog
2. Endpoints for everything, powered by FastAPI
3. Lives in a Docker container for easy deployment and CI/CD
4. Postgres database runs on Google Cloud SQL for even more scaling and accessibility

## What's left?
* There's probably a few things needed to hook up a Discord bot to the endpoints...
* Extra config things in the dockerfile or .yaml file
* Implement a caching layer for faster answers
* Fetch embeddings from OpenAI's endpoint using parallel workers (_without_ incurring ludicrous charges)
* Automatically provision an SQL server in one of the .yaml files to ease setup. I should really also look into seeing how much setup I can automate in the .yamls files...
* Smol boi AI model for fine-tuning the embeddings
* Google sheet that hooks up to said smoll boi to provide training data
* Unit tests & automatic testing
* Some kind of authentication on the endpoints that create new database tables
* Optimize the cleanin regex patterns to get rid of all the moustache placeholders and other formatting BS

## Oh geez that's a lot of things! Good thing I'm doing this project entirely in my free time and have no stakeholders or KPIs to meet 😁😁😁

## How to get it all set up:
Bless your heart for showing interest ❤️ We're still pre-release so this is gonna be rough...
1. Set up your GCP project with:
    * Cloud Run
    * Cloud Build
    * Cloud SQL
    * Secrets Manager
2. In Cloud SQL, start yourself a pretty little postgres DB and make a db user. Put the password in the secrets manager and everything else as raw text in cloudbuilder.yaml
3. In Cloud Build, set up a new job that builds from this repo and run it.
4. Once the thing is built, visit the service URL from Cloud Run and add "/docs" to go to the FastAPI docs screen.
5. Now you can create your articles database! Post a fandom wiki's data dump URL (usually an amazon s3 instance gotten from the wiki's special:statistics page) and watch the program turn that wiki into a database table!
6. That's all I got for now. Cool stuff to come in the future though!!!

## Aknowledgements & Shout-Outs
* Borgeuad et al over at https://arxiv.org/abs/2112.04426 whose RETRO architecture is the basis for this whole damn project.
* OpenAI's cookbook recipe [Question Answering using Embeddings](https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb) whose code I shamelessly stole many times over along with their
* [Collect Wikipedia data about Olympic Games 2020](https://github.com/openai/openai-cookbook/blob/838f000935d9df03e75e181cbcea2e306850794b/examples/fine-tuned_qa/olympics-1-collect-data.ipynb) cookbook recipe as well.
* Kern.ai's blog post [Fine-tuning embeddings for better similarity search](https://dev.to/meetkern/how-to-fine-tune-your-embeddings-for-better-similarity-search-445e) presents a nice walkthrough for fine-tuning embeddings.
