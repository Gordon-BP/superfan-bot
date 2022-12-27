import pandas as pd
import re
import os
from io import StringIO
import sqlalchemy
from dotenv import load_dotenv
import mwparserfromhell
from pyunpack import Archive
from bs4 import BeautifulSoup, ResultSet
import requests
from pathlib import Path
from google.cloud.sql.connector import Connector, IPTypes
import pg8000
import logging as log
from transformers import GPT2TokenizerFast
from .app import get_embedding
logging_client = logging.Client()
logging_client.setup_logging()

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
        pool_size=50,
        max_overflow=10,
        pool_timeout=30,  # 30 seconds
        future=True,
        pool_recycle=1800,  # 30 minutes
        )
    return pool

def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    TODO: make this async or something? Maybe the async comes in how we call this function...

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

def createDataset(url:str, dbPrefix:str, overrideTables:bool=False) -> dict:
    """
    This is supposed to be the main function that:
    1. Fetches the XML data from the provided URL ✅
    2. Parses the data, cleans it, breaks it into chunks, and measures the size ✅
    3. Uploads the data to a table in a Cloud SQL Postgres database ✅
    4. Calls the OpenAI Embeddings endpoint to get embeddings for each chunk 🟨
    5. Saves the embeddings to a table in a Cloud SQL Postgrest database 🟨
    6. Returns a JSON object detailing the table names and sizes 🟨
        TODO: Make it an async process and have some way of monitoring progress
    
    Parameters:
        url(str): The url where the wiki's XML is hosted. Usually an amazon s3 container
        dbPrefix(str): Used in the two tables created: dbPrefix_articles and dbPrefix_embeddings
        overrideTables(bool): If the tables already exist, setting this to True will have the system ovrride it. False by default
    
    Returns
        dict: a JSON object detailing the completed table names and size.
    """
    dbPrefix = str.lower(dbPrefix) # because postgres will die if it sees a capital letter...
    returnDict = {}
    pool = connect_unix_socket()
    insp = sqlalchemy.inspect(pool)
    meta = sqlalchemy.MetaData()
    articles_table = sqlalchemy.Table(f"{dbPrefix}_articles", meta)
    embeddings_table = sqlalchemy.Table(f"{dbPrefix}_embeddings", meta)
    log.info(f"Status for table {articles_table} is {insp.has_table(articles_table, None)}")
    log.info(f"Status for table {embeddings_table} is {insp.has_table(embeddings_table, None)}")
    
    if(not(insp.has_table(articles_table, None)) or overrideTables):
        log.info(f"Creating new tables or overriding existing ones...")
        soup = getWikiAsSoup(url, Path('./data/wiki.xml.7z')) #Fetch the data
        df = cleanData(soup.find_all('page')) #Clean the data
        df = df.drop(df.loc[df['text'].str.contains(r"REDIRECT", re.IGNORECASE)].index).reset_index()
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        df['tokens'] = df.text.apply(lambda x:int(len(tokenizer.encode(x)))) #Get token count to find articles with long subsections
        #TODO make these numbers parameters? Maybe?
        df_short = df.loc[(df.tokens <= 400) & (df.tokens >= 20)]
        df_long = df.loc[df.tokens >= 400]
        chunks = []
        for _, row in df_long.iterrows():
            chunks = chunks + fast_chonk(row, tokenizer)
        df_chunks = pd.DataFrame.from_records(chunks)
        df = pd.concat([df_short, df_chunks])
        df.drop("index", axis=1, inplace=True)
        df.reset_index(drop=True)  
        log.info(f"Articles data created with shape {df.shape}. Now inserting into database....")

        with pool.connect() as conn:
            log.debug("Creating articles table...")
            df.head(0).to_sql(  #drops old table and creates new empty table
                f'{dbPrefix}_articles', conn, 
                if_exists='replace',
                index=False,
                dtype={
                    'id':sqlalchemy.types.INTEGER(),#TODO: is integer the best type here? Or would numeric be better? Does it matter?...
                    'title':sqlalchemy.types.TEXT(),
                    'heading':sqlalchemy.types.TEXT(),
                    'text':sqlalchemy.types.TEXT(),
                    'tokens':sqlalchemy.types.INTEGER()
                }) 
            conn.commit()
        # pg8000 doesn't let sqlalchemy do COPY in a pretty way so we gotta do it like this
        log.debug("Copying articles into articles table...")
        rawConn = pool.raw_connection()
        rawCur = rawConn.cursor()
        output = StringIO()
        df.to_csv(output, sep=',', header=False, index=False, encoding='UTF-8')
        output.seek(0)
        rawCur.execute(f"""
            COPY {dbPrefix}_articles 
            FROM stdin 
            WITH(format csv, DELIMITER ',')""", 
            stream = output)
        rawConn.commit()
        rawConn.close()
  
        log.info(f"Successfully inserted data into table {dbPrefix}_articles")

        os.remove("./data/wiki.xml")
        os.remove("./data/wiki.xml.7z")
        #TODO: Is there a way to clear the memory so we're not keeping the giant articles df around?
        returnDict[0] = {
            "tableName":f"{dbPrefix}_articles",
            "rowCount":df.shape[0],
            "colCount":df.shape[1]
        }
    return returnDict
    # And here's the code that will fetch the embeddings!
    # it's also horribly wrong and slow and just a goddamn mess
    # I will fix it tomorrow!
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
    