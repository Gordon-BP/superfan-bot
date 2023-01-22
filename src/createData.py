import pandas as pd
import re
import os
import numpy as np
import cohere
from dotenv import load_dotenv
import mwparserfromhell
from pyunpack import Archive
from bs4 import BeautifulSoup, ResultSet
import requests
from pathlib import Path
import pinecone
import logging
from transformers import GPT2TokenizerFast
logging.basicConfig(level=logging.DEBUG)
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
load_dotenv()
#log.addHandler(handler)

def getDataAsSoup(url:str, filepath:Path, filename:str='wiki.xml')-> BeautifulSoup:
    """
    Downloads XML data from the supplied URL and turns it into a beautiful soup

    Parameters:
        url(str): The data source URL
        filepath(Path): Where the data will be initially downloaded to
        filename(str): The name of the file used to temporarially store the data
    """
    logging.info(f"Fetching file from {url}...")
    response = requests.get(url)
    if(response.status_code != 200):
        logging.error("Provided URL is not valid")
        raise ValueError 
    else: 
        filepath.write_bytes(response.content)
        Archive(filepath).extractall("./data/")
        soup = BeautifulSoup(open(f"./data/{filename}"), "lxml")
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
              logging.debug(f"Skipping {title} - {heading}")
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
              logging.debug(f"Skipping {title} - {heading}")
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
                    logging.debug(f"Skipping {title} - {subheader} - {heading}")
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
        logging.warn(f"Something went wrong processing {page.title.contents[0]}")
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
    logging.debug(f"Chunking {row.title} - {row.heading}...")
    sentences = row.text.split(". ")
    bigSentence = ""
    chunks = []
    for idx, sentence in enumerate(sentences):
        length = len(tokenizer.encode(sentence))
        if len(tokenizer.encode(bigSentence)) + length <= 512:
            bigSentence = bigSentence + ". " + sentence
        else:
            bigSentence = sentence
            chunks.append({
                "id":int(row.id),
                "title":str(row.title),
                "heading":str(row.heading) + " " + str(idx+1),
                "text": str(bigSentence),
                "tokens": int(len(tokenizer.encode(bigSentence)))
            })

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

def build_embeddings_table(df:pd.DataFrame, dbPrefix:str)-> pd.DataFrame:
    """
    Uses Cohere's embeddings endpoint to embed content and loads it into a new database table.

    Parameters:
        df(pd.DataFrame): The dataframe containing the articles information
        dbPrefix(str): Prefix for this table name
        pool(sqlalchemy.engine.Engine): An sqlalchemy engine to connect to the database with

    Returns:
        pd.DataFrame: The completed embeddings
    """


def create_index(dataSource:str, indexLabel:str, dimension:int = 1024, metric:str='cosine', 
    overrideIndex:bool = False, maxTokens:int = 512, minTokens:int = 10) -> pinecone.Index:
    """
    This is supposed to be the main function that:
    1. Fetches the XML data from the provided URL ✅
    2. Parses the data, cleans it, breaks it into chunks, and measures the size ✅
    3. Calls the embeddings endpoint to get vectors for each data chunk
        3a. I guess the data just chills in-memory for this part?
    4. Uploads the embeddings and content as metadata to Pinecone
    
    Parameters:
        dataSource(str): The url where the wiki's XML is hosted. Usually an amazon s3 container
        indexLabel(str): The name of the pinecone index where data is stores 
        dimension(int): How long your vectors are. Default value is 1024 to match cohere's small model
        metric(str): How to calculate vector similarity. Options include cosine, dotproduct, or euclidean
        overrideIndex(bool): If the index already exists, setting this to True will have the system ovrride it. False by default
        maxTokens(int): The maximum token length for a chunk of text data. Pincone advises 512 so that's the default value
        minTokens(int): The minimum token length for a chunk of text data. 10 by default for no good reason.

    Returns
        pinecone.Index: an Index that can access the pinecone database
    """
    if(not(overrideIndex)):
        logging.info(f"Creating new Index or overriding existing one...")
        # First we need to load and clean the data
        soup = getDataAsSoup(dataSource, Path('./data/wiki.xml.7z')) #Fetch the data
        df = cleanData(soup.find_all('page'))
        df = df.drop(df.loc[df['text'].str.contains(r"REDIRECT", re.IGNORECASE)].index).reset_index()
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        df['tokens'] = df.text.apply(lambda x:int(len(tokenizer.encode(x)))) #Get token count to find articles with long subsections
        df_short = df.loc[(df.tokens <= maxTokens) & (df.tokens >= minTokens)]
        chunks = []
        for _, row in df.loc[df.tokens >= maxTokens].iterrows():
            chunks = chunks + fast_chonk(row, tokenizer)
        df_chunks = pd.DataFrame.from_records(chunks)
        df = pd.concat([df_short, df_chunks])
        df.drop("index", axis=1, inplace=True)
        df.reset_index(drop=True)  
        # limit to 100 rows for testing
        df = df.iloc[:100]
        logging.info(f"Articles data created with shape {df.shape}")
        #TODO: Delete the .xml and .7z files after loading them
        logging.info('aaaaAAAAAAaaaAAAAAA\n\n\n\naaaaaAAAAAaaaaAAaAAAa')
        # Now we embed it!
        # setting it to only embed the first 10 for testing purposes
        logging.info('Getting the Embeddings now....')
        load_dotenv()
        co = cohere.Client(os.environ['COHERE_API_KEY'])
        logging.info("We have the key, now calling cohere...")
        embed = co.embed(
            texts=df['text'],
            model='small',
            truncate='LEFT'
        ).embeddings
        shape = np.array(embed).shape
        logging.debug("Cohere part is done")
        # Start getting Pinecone set up
        logging.info("Making the Pinecone now....")
        load_dotenv()
        pinecone.init(os.environ['PINECONE_API_KEY'], environment='us-west1-gcp')
        # if the index does not exist, we create it
        if indexLabel not in pinecone.list_indexes():
            pinecone.create_index(
                indexLabel,
                dimension=shape[1],
                metric='dotproduct'
            )
        # connect to index
        index = pinecone.Index(indexLabel)
        batch_size = 128
        # Now we start to laod the embeddings....
        ids = df.loc['id'].to_list()
        # create list of metadata dictionaries
        meta = [[{'title': title} for title in df.loc['title']],
        [{'heading': heading} for heading in df.loc['heading']],
        [{'text': text} for text in df.loc['text']]]
        # create list of (id, vector, metadata) tuples to be upserted
        to_upsert = list(zip(ids, embed, meta))

        for i in range(0, shape[0], batch_size):
            i_end = min(i+batch_size, shape[0])
            index.upsert(vectors=to_upsert[i:i_end])

        # let's view the index statistics
        print(index.describe_index_stats())

    return index