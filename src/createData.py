import pandas as pd
import re
import os
import numpy as np
from io import StringIO
import sqlalchemy
from dotenv import load_dotenv
import mwparserfromhell
from pyunpack import Archive
from bs4 import BeautifulSoup, ResultSet
import requests
import logging
from pathlib import Path
import pinecone
import logging as log
from transformers import GPT2TokenizerFast
from .app import get_embedding
log = logging.getLogger("uvicorn.info")
#handler = logging.StreamHandler()
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
#log.addHandler(handler)

def getWikiAsSoup(url:str, filepath:Path)-> BeautifulSoup:
  log.info(f"Fetching file from {url}...")
  response = requests.get(url)
  if(response.status_code != 200):
    log.error("Provided URL is not valid")
    raise ValueError 
  else: 
    filepath.write_bytes(response.content)
    Archive(filepath).extractall("./data/")
    soup = BeautifulSoup(open("./data/wiki.xml"), "lxml")
    return soup

def cleanData(pages: ResultSet, limit:int = -1) -> pd.DataFrame:
  """
    Main method to parse a wiki XML file and turn it into a nice pretty DataFrame like
    | id | Title | Heading | Text |
    |----|-------|---------|------|

  Parameters:
    pages(ResultSet): the collection of pages from the wiki file. Generate using soup.find_all()
    limit(int): Optional, limit the number of pages to iterate through
  
  Returns:
    pd.DataFrame
  """
  dataArr = []
  badHeadings = ['See also', 'References', 'External links', 'Further reading', "Footnotes",
    "Bibliography", "Sources", "Citations", "Literature", "Footnotes", "Notes and references",
    "Photo gallery", "Works cited", "Photos", "Gallery", "Notes", "References and sources",
    "References and notes"]
  counter = 0
  for page in pages:
    if(page.ns.contents[0] == '0'):
      if((limit > 0) & (counter > limit)): break
      try:
        counter +=1
        id = page.id.contents[0]
        title = page.title.contents[0]
        wikipage = mwparserfromhell.parse(
            page.find('text').contents
            )
        sections = wikipage.get_sections(
            flat=True,
            include_lead=True,
            include_headings = True
        )
        for section in sections:
          #TODO should probably replace these lines with some kind of recursive function
          heading = section.filter_headings()
          if(len(heading) == 0):
          # This is a short article with just the summary paragraph
            heading = 'Summary'
            text = section.strip_code(
                normalize=True,
                collapse=True,
                keep_template_params=False)
            if((len(text) < 20) or ("REDIRECT" in text)):
              log.debug(f"Skipping {title} - {heading}")
            dataArr.append({
                  "id":int(id),
                  "title":str(title),
                  "heading":str(heading),
                  "text":str(re.sub(r"\n",'',text))
              })
          elif(len(heading) == 1):
          # This is a properly formatted section
            heading = heading[0].title
            text = section.strip_code(
                normalize=True,
                collapse=True,
                keep_template_params=False)
            # There's a few criteria for whether or not we want to keep this data
            if((heading in badHeadings) or (len(text) < 20) or ("REDIRECT" in text)):
              log.debug(f"Skipping {title} - {heading}")
            else:
              dataArr.append({
                  "id":int(id),
                  "title":str(title),
                  "heading":str(heading),
                  "text":str(re.sub(r"\n",'',text))
              })
          else:
            subheaders = []
            subList = []
            for subsection in section.get_sections():
              subheaders = [x.group() for x in re.finditer(r"={2}([a-zA-Z-']* ?){1,4}={2}\n",str(subsection))]
              subList = re.split(r"={2}([a-zA-Z-']* ?){1,4}={2}\n", str(subsection))
              subList = [str(_) for _ in subList if _!= '']
              subheaders = ['Overview']+[str(_) for _ in subheaders if _!='']
              subsections = zip(subheaders, subList)
              for subheader,subcontent in subsections:
                subsubheaders = [x.group() for x in re.finditer(r"={3}([a-zA-Z-']* ?){1,4}={3}\n",str(subcontent))]
                subsubcontent = re.split(r"={3}([a-zA-Z-']* ?){1,4}={3}\n", str(subcontent))
                subsubheaders = [str(_) for _ in subsubheaders if _!= '']
                subsubcontent = [str(_) for _ in subsubcontent if _!='']
                if len(subsubheaders)==0:
                  heading = re.sub(r"=",'', subheader)
                  text = mwparserfromhell.parse(subsubcontent[0]).strip_code()
                  if((heading in badHeadings) or (len(text) < 20) or ("REDIRECT" in text)):
                    log.debug(f"Skipping {title} - {subheader} - {heading}")
                  dataArr.append({
                    "id":int(id),
                    "title":str(title),
                    "heading":str(heading),
                    "text":str(re.sub(r"\n",'',text))
                  })
                else:
                  subsubsections = zip(subsubheaders, subsubcontent)
                  for subsubsubheader, subsubsubcontent in subsubsections:
                    dataArr.append({
                        "id":int(id),
                        "title":str(title),
                        "heading":str(re.sub(r"=",'', subheader) + " - " + re.sub(r"=",'', subsubsubheader)),
                        "text": str(re.sub("\n",'',mwparserfromhell.parse(subsubsubcontent).strip_code()))
                    })
      except:
        log.warn(f"Something went wrong processing {page.title.contents[0]}")
        counter +=1
  return pd.DataFrame.from_records(dataArr)

def fast_chonk(row:pd.Series, tokenizer:GPT2TokenizerFast) -> list[dict]:
    """
    Super fast content chunker!

    Parameters:
        row(pd.Series): a row of your data whose text needs chunkin'
        tokenizer(GPT2TokenizerFast): the tokenizer, use .from_pretrained("gpt2")
    Returns
        list[dict]: a list of dicts with your new, chunked text. Looks like this:
        [{
            title: Lorem Ipsum,
            heading: Unum 1,
            text: blah blah blah,
            tokens: 79
        }]
    """
    log.debug(f"Chunking {row.title} - {row.heading}...")
    sentences = row.text.split(". ")
    bigSentence = ""
    chunks = []
    for idx, sentence in enumerate(sentences):
        length = len(tokenizer.encode(sentence))
        if len(bigSentence) + length < 400:
            bigSentence = bigSentence + ". " + sentence
        else:
            chunks.append({
                "id":int(row.id),
                "title":str(row.title),
                "heading":str(row.heading) + " " + str(idx+1),
                "text": str(bigSentence),
                "tokens": int(len(tokenizer.encode(bigSentence)))
            })
            bigSentence = sentence

    return chunks

def compute_cost_estimate(df: pd.DataFrame) -> float:
    """
    Makes an estimated cost for fetching embeddings for the entire dataframe. Uses the price for
    ada-embeddigs-text-002 which is 0.0004 USD per 1,000 tokens

    Parameters:
        df(pd.DataFrame): The data you want to tokenize
    Returns:
        float: the estimated cost to embed the whole document
    """
    return (df.tokens.sum()/1000)*0.0004

def build_embeddings_table(df:pd.DataFrame, dbPrefix:str, pool:sqlalchemy.engine.Engine)-> pd.DataFrame:
    """
    Uses Cohere's embeddings endpoint to embed content and loads it into a new database table.

    Parameters:
        df(pd.DataFrame): The dataframe containing the articles information
        dbPrefix(str): Prefix for this table name
        pool(sqlalchemy.engine.Engine): An sqlalchemy engine to connect to the database with

    Returns:
        pd.DataFrame: The completed embeddings
    """


def create_dataset(dataSource:str, indexName:str, dimension:int = 1024, metric:str='cosine', 
    overrideIndex:bool=False) -> pinecone.Index:
    """
    This is supposed to be the main function that:
    1. Fetches the XML data from the provided URL ✅
    2. Parses the data, cleans it, breaks it into chunks, and measures the size ✅
    3. Calls the embeddings endpoint to get vectors for each data chunk
        3a. I guess the data just chills in-memory for this part?
    4. Uploads the embeddings and content as metadata to Pinecone
    
    Parameters:
        dataSource(str): The url where the wiki's XML is hosted. Usually an amazon s3 container
        indexName(str): The name of the pinecone index where data is stores 
        dimension(int): How long your vectors are. Default value is 1024 to match cohere's small model
        metric(str): How to calculate vector similarity. Options include cosine, dotproduct, or euclidean
        overrideIndex(bool): If the index already exists, setting this to True will have the system ovrride it. False by default
    
    Returns
        pinecone.Index: an Index that can access the pinecone database
    """
    pool = connect_unix_socket()
    meta = sqlalchemy.MetaData(bind=pool)
    articles_table = sqlalchemy.Table(f"{dbPrefix}_articles", meta)
    embeddings_table = sqlalchemy.Table(f"{dbPrefix}_embeddings", meta)
    log.info(f"Status for articles table {articles_table.name} is {articles_table.exists()}")
    log.info(f"Status for embeddings table {embeddings_table.name} is {embeddings_table.exists()}")
    log.info(f"Override tables is {overrideTables}")
    if(not(articles_table.exists()) or overrideTables):
        log.info(f"Creating new articles table or overriding existing one...")
        articles_df = build_articles_table(url, pool, dbPrefix)
       # log.info("Using existing articles table...")
       # dumbPool = connect_unix_socket(future=False)
       # articles_df = pd.read_sql_query(f"SELECT * FROM {dbPrefix}_articles", dumbPool)
        log.info(f"Creating new embeddings table or overriding existing one...")
        embeddings_df = build_embeddings_table(articles_df, dbPrefix, pool)
        # Build a nice little JSON object to return
    else:
        log.info("Tables already exist, loading them into memory....")
        with pool.connect() as conn:
            articles_df = pd.read_sql_table(f"{dbPrefix}_articles", conn)
            embeddings_df = pd.read_sql_table(f"{dbPrefix}_embeddings", conn)
            embeddings_df['vec'] = embeddings_df.vec.apply(lambda x: [float(i) for i in x[1:-1].split(",")])

    return {
            f"{dbPrefix}_articles":{
                "tableStatus":articles_table.exists(),
                "dataFrame": articles_df
        },
            f"{dbPrefix}_embeddings":{
                "tableStatus":embeddings_table.exists(),
                "dataFrame":embeddings_df
            }
        }