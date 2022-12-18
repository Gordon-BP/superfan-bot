# superfan-bot
A conversational AI that reads a fandom's wiki and becomes an expert for your Discord server!

Big shoutouts to OpenAI, Kern AI, and for their amazing sample code:
* [Question Answering using Embeddings](https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb)
* [Collect Wikipedia data about Olympic Games 2020](https://github.com/openai/openai-cookbook/blob/838f000935d9df03e75e181cbcea2e306850794b/examples/fine-tuned_qa/olympics-1-collect-data.ipynb)
* [Fine-tuning embeddings for better similarity search](https://github.com/code-kern-ai/refinery-sample-projects/tree/finetuning-similarity-search)



## Quick Start
1. In `apiKeys.py` add:
    * Your OpenAI API key
    * The XML file containing your wiki's info
    * A filename for your embeddings
    * The URI that gets prepended to all the tags in the XML file
2. TODO: dockerize this app so that people can just run the container.
3. Should probably manually label a bit of data to fine-tune the model. I hear a little goes a long way!
4. Once the docker stuff is runnning, you have to auth with Discord somehow idk
5. Talk to your bot on discord and have fun~~