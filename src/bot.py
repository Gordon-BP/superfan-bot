import discord
from discord.ext import commands
import dotenv
import cohere
from createData import getDataAsSoup, cleanData, fast_chonk, create_index, create_index
import os
import sys
from pathlib import Path
from app import get_embedding, query_index, prompt_completion
import logging
import pandas as pd
import re
from transformers import GPT2TokenizerFast

# Init the bot with the right discord permissions
dotenv.load_dotenv()
token = os.environ["DISCORD_TOKEN"]
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='$', intents=intents)

# Just fot testing purposes, we're going to hard-code a dataset
df = pd.read_csv("./data/Witcher_preview.csv")

# Friendly startup message
@bot.event
async def on_ready():
    # Let's set up logging!
    discord.utils.setup_logging(level=logging.INFO, root=False)
    global botLogger
    botLogger = logging.getLogger(f"{bot.user}")
    c_handler = logging.StreamHandler()
    c_formatter = logging.Formatter('%(asctime)s %(levelname)s\t%(name)s\t%(message)s')
    c_handler.setFormatter(c_formatter)
    botLogger.addHandler(c_handler)
    botLogger.setLevel(logging.INFO) # <-- THIS!

    # Next let's initialize cohere!
    if(os.environ['COHERE_API_KEY']):
        global co 
        co = cohere.Client(os.environ['COHERE_API_KEY'])

    botLogger.info(f'The bot is ready!')

# Command just to test if the bot is onling
@bot.command()
async def repeat(ctx: discord.ext.commands.context.Context , *args):
    botLogger.info('User requested repeat: {}'.format(' '.join(args)))
    await ctx.send('{}'.format(' '.join(args)))

@bot.command()
async def create_dataset(ctx: discord.ext.commands.context.Context, url:str, index:str, **kwargs):
    """
    Takes a datasource URL and creates a Pinecone Index vector database out of it!
    """
    ctx.send(f'Creating an index for {url}...')
    # First, let's type check all of our arguments!
    if not(isinstance(url, str)):
        await ctx.send(f"The URL is {type(url)} not str")
        botLogger.error(f"Invalid Parameter: The URL is {type(url)} not str")
        return
    elif not(isinstance(index, str)):
        await ctx.send(f"The Index label is type {type(index)} and not a valid string")
        botLogger.error(f"Invalid Parameter: The Index label is {type(url)} not str")
        return
    else:
        botLogger.info("Creating index, please wait")
        #TODO: this is where we unpack and verify the kwargs
        index = create_index(url, index, overrideIndex=False)
        await ctx.send('Index created! Here are the stats:')
        stats = index.describe_index_stats()
        await ctx.send(str(stats))
    
@bot.command()
async def embed(ctx, *args):
    text = str(' '.join(args))
    botLogger.info(text)
    try:
        result = get_embedding(text)
        size = len(result)
        await ctx.send(f"Here's a preview of the the embeddings for {text}, the real one is {size} vectors long.\n{result[1:10]}")
    except Exception as e:
        botLogger.error(f"There was an error: {e}")
        tb = sys.exc_info()[2]
        botLogger.error(e.with_traceback(tb))
@bot.command()
async def run_test(ctx):
    botLogger.info("Starting test....")
    botLogger.info(f'\n{df.columns}')
    texts = df['text'].to_list()
    botLogger.info(f"Preview of df['text']:\n{texts[0:10]}")
    results = co.embed(
        texts=texts,
        model='small',
        truncate='LEFT'
    ).embeddings
    botLogger.info(f"Here's a preview of the embeddings: nope nvm")


@bot.command()
async def query(ctx:discord.ext.commands.context.Context, index:str, *args):
    query = ' '.join(args)
    results = query_index([query], index)
    results_list = [match['metadata']['text'] for match in results['matches']]
    bot_answer = prompt_completion(query, results_list)
    await ctx.send(f"Here's what I think the answer to '{query}' is:\n{bot_answer}")

bot.run(token)