import discord
from discord.ext import commands
import dotenv
import cohere
from createData import create_index
import os
import logging

# Init the bot with the right discord permissions
dotenv.load_dotenv()
token = os.environ["DISCORD_TOKEN"]
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='$', intents=intents)


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
        index = await create_index(url, index)
        await ctx.send('Index created! Here are the stats:')
        stats = index.describe_index_stats()
        await ctx.send(str(stats))
    

bot.run(token)