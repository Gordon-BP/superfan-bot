# superfan-bot-v2
Turn your favorite fandom wiki into a conversational AI!
Also thinking about changing the name to something catchier. Possibilities include "Wiki-speaks"

## Back for round two, this time 100% lower running costs
When I was last developing this, I ate up my whole USD 20 budget on dumb Cloud Run and Cloud SQL services. Sure I learned a lot, but that was _twenty bucks!_ Round two has three big goals: Free, Easy, and Functional.

### Free
* Goodbye Google Cloud Run, you cost money and I'm not about that. This time, we're sticking with a free, f1 micro Cloud VM to run our container on and that's that!
* Google Cloud SQL and Cloud Storage are also going away, this time being replaced by a free managed (!) database from Pinecone
* OpenAI's embeddings are great and they let me market this project as chatGPT, but _they cost money._ Now I'm a be switching to Co:here and their endpoints, with a possibility of adding options for OpenAI's big money premium API in the future.

### Easy
* None of this fancy auto-deploy every time my github repo is updated. If I want to change the version running on the server I'll just manuyally do that tyvm. 
* PostgreSQL server was slow and hard to set up, especially Google Cloud SQL (which cost money, too). Pinecone is not only free but specialized for vector data and will handle the similarity searching, too!

### Functional
* This time I'm going to make the Discord bot part I swear! 
* I love FastAPI but I'm dropping that so everything can be done solely through Discord.

## To-Do List:
- [x] Pull all the wiki data and use it to populate the database, not just a locall-hosted sample (I sample it anyways because otherwise it would take a long time to embed everything)
- [x] Memory management! Delete the XML files after embedding finishes
- [x] Get the docker image properly up and running 
- [x] Configure the Google cloud VM to auto-start the discord bot and take the Discord token as an env variable
- [x] Create commands for adding Cohere and Pinecone API keys through Discord
- [ ] (Do we always need to specify the index name when we query the index?)
- [ ] Do a better job parsing the wiki data. Smaller chunks, no code, no redirects, no bullshit.
- [ ] Experiment with different prompt screens and hyper parameters on the Cohere generate endpoint
- [ ] Add optional code to interface with OpenAI endpoints (for people with $$$)
- [ ] Make a cool github.io webpage with how-to information and gifs showing how cool this bot is
- [ ] Feature to add custom data to the bot database via Google Sheet (for small talk or other stuff)

## Aknowledgements & Shout-Outs
* Big props to [Pinecone](https://pinecone.io) and [Cohere](https://cohere.ai) for having generous free tiers!
* Borgeuad et al over at https://arxiv.org/abs/2112.04426 whose RETRO architecture is the basis for this whole damn project.
* OpenAI's cookbook recipe [Question Answering using Embeddings](https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb) whose code I shamelessly stole many times over along with their
* [Collect Wikipedia data about Olympic Games 2020](https://github.com/openai/openai-cookbook/blob/838f000935d9df03e75e181cbcea2e306850794b/examples/fine-tuned_qa/olympics-1-collect-data.ipynb) cookbook recipe as well.
* The nice people behind Pinecone's documentation whose code I stole to [create and store embeddigns from Cohere](https://docs.pinecone.io/docs/cohere).
* Kern.ai's blog post [Fine-tuning embeddings for better similarity search](https://dev.to/meetkern/how-to-fine-tune-your-embeddings-for-better-similarity-search-445e) presents a nice walkthrough for fine-tuning embeddings.
