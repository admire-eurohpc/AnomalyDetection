from elasticsearch import Elasticsearch
import pandas as pd
from tqdm import tqdm

def query_elastic(index_name: str) -> list:
    
    res = es.count(index=index_name)
    size = res['count']


    body = { "size": 10000,
                "query": {
                    "match_all" : {}
                },
                "sort": [
                    {"@timestamp": "asc"},
                    {'hostname.keyword': "desc"}
                ]
            }

    
    result = es.search(index=index_name, body=body)
    bookmark = [result['hits']['hits'][-1]['sort'][0], str(result['hits']['hits'][-1]['sort'][1]) ]
    
    while len(result['hits']['hits']) < size:
        res = es.search(index=index_name, body=body)
        print(len(res['hits']['hits']))
        for el in res['hits']['hits']:
            result['hits']['hits'].append(el)
        
        if size == len(result['hits']['hits']):
            break
        
        bookmark = [res['hits']['hits'][-1]['sort'][0], str(result['hits']['hits'][-1]['sort'][1])]
        print(len(result['hits']['hits']), bookmark)

        if size - len(result['hits']['hits']) < 10000:
            body_size = size - len(result['hits']['hits'])
            print(body_size + len(result['hits']['hits']), size, body_size)
            body = {"size": body_size,
                    "query": {
                        "match_all" : {}
                    },
                    "search_after": bookmark,
                    "sort": [
                        {"@timestamp": "asc"},
                        {'hostname.keyword': "desc"}
                    ]
                    }
        else:
            body = {"size": 10000,
                    "query": {
                        "match_all" : {}
                    },
                    "search_after": bookmark,
                    "sort": [
                        {"@timestamp": "asc"},
                        {'hostname.keyword': "desc"}
                    ]
                }

    return result['hits']['hits']

def elastic_to_parquet(index_name: str):
    
    elastic_data = query_elastic(index_name=index_name)

    docs = pd.DataFrame()
    res_list = []
    # iterate each Elasticsearch doc in list
    print("ncreating objects from Elasticsearch data.")
    for _, doc in tqdm(enumerate(elastic_data), total=len(elastic_data)): 
        # get _source data dict from document
        source_data = doc["_source"]
        # get _id from document
        _id = doc["_id"]
        # create a Series object from doc dict object
        doc_data = pd.Series(source_data, name=_id).to_frame().T
        res_list.append(doc_data)

    docs = pd.concat(res_list, axis=0)

    print(docs.info())
    docs.to_parquet(f"data/eagle_nodes/{index_name}.parquet")

if __name__ == "__main__":

    #Connect to Elastic
    es = Elasticsearch([{"host": "10.200.132.195", "port": 9200, "scheme": "http"}], basic_auth=('user', 'password'))

    if not es.ping():
        raise ValueError("Connection failed")

    indices_list = es.indices.get_alias(index="*eagle_nodes*").keys()
    for elem in indices_list : 
        elastic_to_parquet(index_name = elem)
