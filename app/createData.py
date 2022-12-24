import pandas as pd
from bs4 import BeautifulSoup
import re
import os
import sqlalchemy
import ps8000
import mwparserfromhell
from pyunpack import Archive
from bs4 import BeautifulSoup, ResultSet
import pandas as pd
import requests
from pathlib import Path
from transformers import GPT2TokenizerFast
from nltk.tokenize import sent_tokenize

# probably don't need these two
from app import get_embedding
from apiKeys import XML_FILEPATH, EMBEDDINGS_FILEPATH, URI

def getWikiAsSoup(url:str, filepath:Path)-> BeautifulSoup:
  response = requests.get(url)
  filepath.write_bytes(response.content)
  Archive(filepath).extractall("/content/")
  return BeautifulSoup(open("wiki.xml"), "lxml")

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
              subList = [_ for _ in subList if _!= '']
              subheaders = ['Overview']+[_ for _ in subheaders if _!='']
              subsections = zip(subheaders, subList)
              for subheader,subcontent in subsections:
                subsubheaders = [x.group() for x in re.finditer(r"={3}([a-zA-Z-']* ?){1,4}={3}\n",str(subcontent))]
                subsubcontent = re.split(r"={3}([a-zA-Z-']* ?){1,4}={3}\n", str(subcontent))
                subsubheaders = [_ for _ in subsubheaders if _!= '']
                subsubcontent = [_ for _ in subsubcontent if _!='']
                if len(subsubheaders)==0:
                  heading = re.sub(r"=",'', subheader)
                  text = mwparserfromhell.parse(subsubcontent[0]).strip_code()
                  if((heading in badHeadings) or (len(text) < 20) or ("REDIRECT" in text)):
                    print(f"Skipping {title} - {subheader} - {heading}")
                  dataArr.append({
                      "title":title,
                      "heading":heading,
                      "text":text 
                  })
                else:
                  subsubsections = zip(subsubheaders, subsubcontent)
                  for subsubsubheader, subsubsubcontent in subsubsections:
                    dataArr.append({
                        "title":title,
                        "heading":re.sub(r"=",'', subheader) + " - " + re.sub(r"=",'', subsubsubheader),
                        "text": mwparserfromhell.parse(subsubsubcontent).strip_code()
                    })
      except:
        print(f"Error processing page {page.title.contents[0]}")
        counter +=1
  return pd.DataFrame.from_records(dataArr)

def reduce_long(
    dataSection: pd.Series,  max_len: int = 590
) -> pd.DataFrame:
    """
    Reduce a long text to a maximum of `max_len` tokens by potentially cutting at a sentence end
    """
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    if not long_text_tokens:
        long_text_tokens = len(tokenizer.encode(long_text))
    if long_text_tokens > max_len:
        sentences = sent_tokenize(long_text.replace("\n", " "))
        ntokens = 0
        for i, sentence in enumerate(sentences):
            ntokens += 1 + len(tokenizer.encode(sentence))
            if ntokens > max_len:
                return ". ".join(sentences[:i][:-1]) + "."
    if dataSection.tokens <= max_len:
        return pd.DataFrame(dataSection)
    else:
        dataArr = []
        sentences = sent_tokenize(dataSection.text.str.replace("\n", " "))
        ntokens = 0
        counter = 0
        last_idx = 0
        for i, sentence in enumerate(sentences):
            ntokens += 1 + len(tokenizer.encode(sentence))
            if ntokens > max_len:
                counter += 1
                last_idx=i
                dataArr.append({
                    "id":dataSection.id,
                    "title":dataSection.title,
                    "heading":dataSection.heading + " " + str(counter),
                    "text":". ".join(sentences[last_idx:i])
                    })
        return pd.DataFrame.from_records(dataArr)

def connect_unix_socket() -> sqlalchemy.engine.base.Engine:
    # Note: Saving credentials in environment variables is convenient, but not
    # secure - consider a more secure solution such as
    # Cloud Secret Manager (https://cloud.google.com/secret-manager) to help
    # keep secrets safe.
    db_user = os.environ["DB_USER"]  # e.g. 'my-database-user'
    db_pass = os.environ["DB_PASS"]  # e.g. 'my-database-password'
    db_name = os.environ["DB_NAME"]  # e.g. 'my-database'
    unix_socket_path = os.environ["INSTANCE_UNIX_SOCKET"]  # e.g. '/cloudsql/project:region:instance'

    pool = sqlalchemy.create_engine(
        # Equivalent URL:
        # postgresql+pg8000://<db_user>:<db_pass>@/<db_name>
        #                         ?unix_sock=<INSTANCE_UNIX_SOCKET>/.s.PGSQL.5432
        # Note: Some drivers require the `unix_sock` query parameter to use a different key.
        # For example, 'psycopg2' uses the path set to `host` in order to connect successfully.
        sqlalchemy.engine.url.URL.create(
            drivername="postgresql+pg8000",
            username=db_user,
            password=db_pass,
            database=db_name,
            query={"unix_sock": "{}/.s.PGSQL.5432".format(unix_socket_path)},
            pool_pre_ping=True
        ),
        # ...
    )
    return pool

def createDatabaseTable(tableName:str) -> str:
    """
    TODO: Add a proper docstring once I figure out how these goddamn databases work
    """
  # First, download and prep the data
    url = 'https://s3.amazonaws.com/wikia_xml_dumps/c/ci/civilization_pages_current.xml.7z'
    soup = getWikiAsSoup(url, Path('wiki.xml.7z'))
    df = cleanData(soup.find_all('page'))
    df = df.drop(df.loc[df['text'].str.contains("REDIRECT")].index).reset_index()

    # Next, upload it to our PostgreSQL database
    pool = connect_unix_socket()
    try:
        with pool.connect as db_conn:
            df.to_sql(tableName, db_conn)
        return f"Successfully created table {tableName}"
    except:
        return "There was an error, check the logs?"
        

def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
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
    return df.token.sum()/1000*0.0004

def createDataset():
    uri = URI
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    print("Beginning data parsing...")
    sectioned_pages_df = []
    print("Extracting content pages...")
    content_df = getContentPages(uri, XML_FILEPATH)
    print("Breaking content df into sections...")
    #TODO- try to optimize this and remove the iterrows(), this is a bottleneck
    for _, row in content_df.iterrows():
        sectioned_pages_df.append(break_into_sections(row))

    df = pd.concat(sectioned_pages_df, ignore_index=True)
    print("Tokenizing sections...")
    df['tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
    
    # Lets clean this dataset even more. 
    # First, drop all the tiny little article stubs
    print("Cleaning dataset...")
    bad_headers = ['Videos', 'See also', 'References', 'XML', 'Python', 'External links', '\n\n']
    df.drop(df[df.tokens < 30].index, axis=0, inplace=True)
    df.drop(df[df.tokens > 400 ].index, axis=0, inplace=True)
    df.drop(df[df.header.str.strip().isin(bad_headers)].index, axis=0, inplace=True)

    print(df.head())
    print("Saving to file....")
    df.to_csv("data/Civ6-Atricles-Cleaned.csv")

    if(exists(EMBEDDINGS_FILEPATH)):
        print("Embeddings file already found, skipping embeddings calls")
    else:
        cost = compute_cost_estimate(df)
        print(f"Fetching embeddings for this document is estimated to cost USD: {cost}")
        embeddings = compute_doc_embeddings(df)
        context_df = pd.DataFrame.from_dict(embeddings)
        context_df.to_csv("data/raw_embeddings.csv")
        embeddings_dict = []
        for idx, row in df.iterrows():
            mylist = [row.title, row.header] + embeddings[idx]
            embeddings_dict.append(mylist)
        edf = pd.DataFrame.from_records(embeddings_dict)
        edf.to_csv(EMBEDDINGS_FILEPATH)

if __name__ == '__main__':
    createDataset()