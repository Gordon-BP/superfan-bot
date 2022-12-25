import pandas as pd
import re
import os
import sqlalchemy
from dotenv import load_dotenv
import mwparserfromhell
from pyunpack import Archive
from bs4 import BeautifulSoup, ResultSet
import requests
from pathlib import Path
from google.cloud.sql.connector import Connector, IPTypes
import pg8000
from transformers import GPT2TokenizerFast
from .app import get_embedding

def getWikiAsSoup(url:str, filepath:Path)-> BeautifulSoup:
  response = requests.get(url)
  if(requests.status != 200):
    print("Provided URL is not valid")
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
              print(f"Skipping {title} - {heading}")
            dataArr.append({
                  "id":id,
                  "title":title,
                  "heading":heading,
                  "text":text
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
              print(f"Skipping {title} - {heading}")
            else:
              dataArr.append({
                  "id":id,
                  "title":title,
                  "heading":heading,
                  "text":text
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
                    print(f"Skipping {title} - {subheader} - {heading}")
                  dataArr.append({
                      "id":id,
                      "title":title,
                      "heading":heading,
                      "text":text 
                  })
                else:
                  subsubsections = zip(subsubheaders, subsubcontent)
                  for subsubsubheader, subsubsubcontent in subsubsections:
                    dataArr.append({
                        "id":id,
                        "title":title,
                        "heading":re.sub(r"=",'', subheader) + " - " + re.sub(r"=",'', subsubsubheader),
                        "text": mwparserfromhell.parse(subsubsubcontent).strip_code()
                    })
      except:
        print(f"Error processing page {page.title.contents[0]}")
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
            tokens: 
        }]
    """
    print(f"Chunking {row.title} - {row.heading}...")
    sentences = row.text.split(". ")
    bigSentence = ""
    chunks = []
    for idx, sentence in enumerate(sentences):
        length = len(tokenizer.encode(sentence))
        if len(bigSentence) + length < 400:
            bigSentence = bigSentence + ". " + sentence
        else:
            chunks.append({
                "id":row.id,
                "title":row.title,
                "heading":str(row.heading) + " " + str(idx+1),
                "text": bigSentence,
                "tokens": len(tokenizer.encode(bigSentence))
            })
            bigSentence = sentence

    return chunks

def connect_unix_socket() -> sqlalchemy.engine.base.Engine:
    """
    Connects to a Google Cloud Postgresql database. 
    Make sure that the env variables are properly defined in your cloudbuilder.yaml file.
    """
    # Uncomment the below for local development
    # Note: These env variables are defined in cloudbuilder.yaml
    # Except for the password, you have to make that in the secret manager
    # Cloud Secret Manager (https://cloud.google.com/secret-manager)
    load_dotenv() # this is only good for local development
    db_user = os.environ["DATABASE_USER"]  # e.g. 'my-database-user'
    db_pass = os.environ["DATABASE_PASS"]  # e.g. 'my-database-password'
    db_name = os.environ["DATABASE_NAME"]  # e.g. 'my-database'
    instance_connection_name = os.environ["INSTANCE_CONNECTION_NAME"]  # e.g. '/cloudsql/project:region:instance'
    ip_type = IPTypes.PUBLIC
    #Let's make sure folks actually set their variables in the yaml file...
    assert db_user != "TEST_USER_CHANGE_ME", "Invalid database username in cloudbuilder.yaml"
    assert db_pass != "hunter2", "Invalid database password in cloudbuilder.yaml"
    assert db_name != "DB_NAME_GOES_HERE", "Invalid database name in cloudbuilder.yaml"
    assert instance_connection_name != "PROJECT:REGION:INSTANCE", "Invalid socket path in cloudbuilder.yaml"
    
    connector = Connector()
    def getconn() -> pg8000.dbapi.Connection:
        conn: pg8000.dbapi.Connection = connector.connect(
            instance_connection_name,
            "pg8000",
            user=db_user,
            password=db_pass,
            db=db_name,
            ip_type=ip_type,
        )
        return conn
    pool = sqlalchemy.create_engine(
        "postgresql+pg8000://",
        creator=getconn,
        # [START_EXCLUDE]
        # Pool size is the maximum number of permanent connections to keep.
        pool_size=5,
        # Temporarily exceeds the set pool_size if no connections are available.
        max_overflow=2,

        # 'pool_timeout' is the maximum number of seconds to wait when retrieving a
        # new connection from the pool. After the specified amount of time, an
        # exception will be thrown.
        pool_timeout=30,  # 30 seconds

        # 'pool_recycle' is the maximum number of seconds a connection can persist.
        # Connections that live longer than the specified amount of time will be
        # re-established
        pool_recycle=1800,  # 30 minutes
        )
    return pool

def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    TODO: make this async or something

    Parameters:
        df(pd.DataFrame): The data to embed. Should look like:
            | id | title | heading | text | tokens |
            |----|-------|---------|------|--------|
    Returns:
        dict: [(tite, heading), 1526 numbers]
    """
    return {
        idx: get_embedding(r.text.replace("\n", " ")) for idx, r in df.iterrows()
    }

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

def createDataset(url:str, dbPrefix:str) -> str:
    """
    OK so this is supposed to be the main def that brings everything together. The way it should work is:
    1. Connectes to the goddamn database
    2. Checks to see if the articles table is already there
        TODO: Is there a way we can add metadata to the table to label the articles with their source?
    3. Check to see if the embeddings table is already there
        TODO: Is there a way we can link this embeddings table to the articles table?
    4. If the tables exist and everything is good we should return 
    5. If there's no data we gotta build it all from scratch, including:
        a) Fetch the data and unpack it
        b) Parse the XML and extract the content
        c) Clean the content, chunk larger paragraphs, and put it in the articles table
        d) Get embeddings for all the data (thanks OpenAI!)
        e) Save the embeddings to the embeddings table
    6. The embeddings is going to take a long time. 
        TODO: Make it an async process and have some way of monitoring progress
    7. TODO: What should we return?
    """
    print("First let's connect to the database")
    pool = connect_unix_socket()
    insp = sqlalchemy.inspect(pool)
    meta = sqlalchemy.MetaData()
    
    print("Next let's see if our tables are already there")
    articles_table = sqlalchemy.Table(f"{dbPrefix}_articles", meta)
    embeddings_table = sqlalchemy.Table(f"{dbPrefix}_embeddings", meta)
    if(~insp.has_table(articles_table, None)):
    # Fetch the data from the url, parse it, clean it, chunk it, and pop it into a SQL table
        print(f"Fetching data from {url}...")
        soup = getWikiAsSoup(url, Path('./data/wiki.xml.7z'))
        
        print("Clening data...")
        df = cleanData(soup.find_all('page'))
        df = df.drop(df.loc[df['text'].str.contains(r"REDIRECT", re.IGNORECASE)].index).reset_index()
        #TODO: I'm sure there's a better way to do this other than iterrows
        dfArray = []
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        
        print("Tokenizing data...")
        df['tokens'] = df.text.apply(lambda x:len(tokenizer.encode(x)))
       
        print("Chunking long passages...")
        df_short = df.loc[(df.tokens <= 400) & (df.tokens >= 20)]
        df_long = df.loc[df.tokens >= 400]
        chunks = []
        for _, row in df_long.iterrows():
            chunks = chunks + fast_chonk(row, tokenizer)
        df_chunks = pd.DataFrame.from_records(chunks)
        df = pd.concat([df_short, df_chunks])
        
        print(f"Final data created with shape {df.shape}. Now inserting into database....")
        #Yeet the rows of 20 tokens or fewer as those won't contain enough data to be useful
        print("Pushing data to a new table....")
        df.iloc[0:1000].to_sql(f"{dbPrefix}_articles", pool, if_exists='append', index=False, chunksize=100)
        
        print(f"Successfully created table {dbPrefix}_articles")
        #TODO: See if these two lines actually work or if they fuck everything up
        os.remove("./data/wiki.xml")
        os.remove("./data/wiki.xml.7z")
        return "Success? Check your database!"

"""  if(~insp.has_table(embeddings_table, None)):
    # Time to make the embeddings babyyyy~~~
        try:
            cost = compute_cost_estimate(df)
            print(f"Fetching embeddings for this document is estimated to cost USD: {cost}")
            #TODO: Maybe put a way to stop this if the cost is too high?...
            #TODO: Make a joke mode where we add $100 to the cost 
            embeddings = compute_doc_embeddings(df)
            embeddings_dict = []
            for idx, row in df.iterrows():
                mylist = [row.title, row.header] + embeddings[idx]
                embeddings_dict.append(mylist)
            with pool.connect() as db_conn:
                edf = pd.DataFrame.from_records(embeddings_dict)
                edf.to_sql(f"{dbPrefix}_embeddings")
                print(f"Successfully created table {dbPrefix}_embeddings")
        except:
            print("Something went wrong creating the embeddings table. Check the logs")
            return "Something went wrong with the embeddings table. Check the logs"
  """  