from discord import Intents, utils
from discord.ext import commands
import dotenv
import os
import logging

# Init the bot with the right discord permissions
dotenv.load_dotenv()
token = os.environ["DISCORD_TOKEN"]
intents = Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='$', intents=intents)


# Friendly startup message
@bot.event
async def on_ready():
    utils.setup_logging(level=logging.INFO, root=False)
    global botLogger
    botLogger = logging.getLogger(f"{bot.user}")
    c_handler = logging.StreamHandler()
    c_formatter = logging.Formatter('%(asctime)s %(levelname)s\t%(name)s\t%(message)s')
    c_handler.setFormatter(c_formatter)
    botLogger.addHandler(c_handler)
    botLogger.setLevel(logging.INFO) # <-- THIS!

    botLogger.info(f'The bot is ready!')

# Command just to test if the bot is onling
@bot.command()
async def repeat(ctx, *args):
    botLogger.info('User requested repeat: {}'.format(' '.join(args)))
    await ctx.send('{}'.format(' '.join(args)))



bot.run(token)