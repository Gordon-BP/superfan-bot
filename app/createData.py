import pandas as pd
from bs4 import BeautifulSoup
import re
from os.path import exists
from transformers import GPT2TokenizerFast
from app import get_embedding
from apiKeys import XML_FILEPATH, EMBEDDINGS_FILEPATH, URI

def cleanText(rawText:str) -> str:
    """
    Cleans raw text from XML using regex magic.
    TODO Refine the crap out of this function so that it cleans better
    
    Parameters:
        rawText(str): The text to clean
    
    Returns
        str: Cleaned version of text
    """
    cleanText = re.sub(r"\|\S* ?\S* ?\S* ?\S*\]\]",'',rawText)
    cleanText = re.sub("\]+|\[+",'',cleanText)
    cleanText = re.sub(r"'+", '', cleanText)
    cleanText = re.sub(r"{{\S* ?\S* ?\S*\|\S* ?\S* ?\S* ?\S* ?}}",'',cleanText)
    #cleanText = re.sub(r"{{(.*)",'',cleanText)
    return cleanText

def getContentPages(uri:str, xmlFilepath:str, namespace:str = "0", limit:int = 1000)-> pd.DataFrame:
    """
    Parses the XML file to extract all the content pages into a DataFrame.

    Parameters:
        uri(str): The XML URI string that's prepended onto all the tags. Usually a URL like "mediawiki.org"
        xmlFile(str): The path to the exported XML file
        namespace(str): The namespace representing the content category. '0' by default
        limit(int): Maximum number of content pages to extract. 1000 by default
    Returns:
        pd.DataFrame of the content in three columns:
        | id | title | text |
        |----|-------|------|
    """
    soup = BeautifulSoup(open("civilization_pages_current.xml"), "lxml")
    pages = soup.find_all('page')
    contentPages = []
    for page in pages:
        try:
            id = page.id.contents[0]
            title = page.title.contents[0]
            text = re.sub(r"\n",'',
                re.sub(r"{{PAGENAME}}", title,page.find('text').contents[0]))
            text = cleanText(text)
            if(len(text) < 50):
                continue
            else:
                contentPages.append({"id":id, "title":title, "text":text})
                print(f"Extracted page: {title}")
                if(len(contentPages) > limit):
                    break
        except:
            print(f"Error parsing page {page.title}")
        continue
    print("Let's put all these guys into a DataFrame now!")
    df =  pd.DataFrame.from_records(contentPages)
    return df

def break_into_sections(row: pd.Series) -> pd.DataFrame:
    """
    Takes an entire article and turns it into multiple rows of a DataFrame, each labeled by section.

    Parameters:
        row(pd.Series): The row with a wiki article in it
    
    Returns:
        pd.DataFrame of the row broken into multiple rows by section like:
        | id |  title  |  header | text |
        |----|---------|---------|------|
        | 12 | Victory | Summary | .... |

        If the article is super short and has no headers, a default header 'Summary' will be applied.
    
    """
    rows = []
    # Not sure if all wiki XML files denote headers like ' === Header Title==='
    # If they don't, this is the regex to change
    headers = re.findall(r"={2,3}([a-zA-Z\s]*)===?", row.text)

    # This is the logic that applies a default header
    if(len(headers) == 0):
        rows.append({
                "id":row.id,
                "title":row.title,
                "header":"Summary",
                "text":row.text
            })
    else:
        headerIdx = [0]
        for header in headers:
            headerStart = re.search(f"{header}", row.text)
            if headerStart is not None:
                headerIdx.append(headerStart.span()[0]) 
            else:
                headers.remove(header)
        # Gotta pop 'Summary' in for the first section of longer articles, too.
        # And add the length to the index list so we can properly capture the last section.
        headers.insert(0,"Summary")
        headerIdx.append(len(row.text))

        for i in range(len(headers)):
            try:
                text = row.text[headerIdx[i]: headerIdx[i+1]]
                rows.append({
                    "id":row.id,
                    "title":row.title,
                    "header":headers[i],
                    "text":text
                })
            except:
                print(f"Something went wrong trying to parse {headers[i]} in {row.title}")
    return pd.DataFrame.from_records(rows)

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