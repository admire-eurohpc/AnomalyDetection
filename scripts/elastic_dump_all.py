from elasticsearch import Elasticsearch
import pandas as pd
import sys, time, io, os
start_time = time.time()

sys.path.append('../../')

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))


es = Elasticsearch([{"host": "10.200.132.195", "port": 9200, "scheme": "http"}]) #need VPN connection

es_index = "eagle_temps-2023-01" #index name

#consider changing this parameter to chassis if you experiance OOM in this script. Chassis divide data into 4 similar chunks instead of taking all at once. 
#If you decide to go with chassis, change all "kiosk" to "chassis". Chassis has values from 1 to 4.
kiosk = 6 #taking kiosk 6 temps, 96% od data comes from kiosk 6, the rest is probably from some smaller machines

body2 = {
        "query": {
        "term" : { "kiosk" : kiosk }
            }
        }

res = es.count(index=es_index, body= body2)
size = res['count']
print(size)
#size = 19740000 #set value by hand, in other case for loop doesn't work
print(size)


body = { "size": 10,
            "query": {
                "term" : {
                    "kiosk" : kiosk
                }
            },
            "sort": [
                {"date": "asc"},
                {'power': "desc"}
            ]
        }

result = es.search(index=es_index, body= body)
bookmark = [result['hits']['hits'][-1]['sort'][0], str(result['hits']['hits'][-1]['sort'][1]) ]

body1 = {"size": 10,
            "query": {
                "term" : {
                    "kiosk" : kiosk
                }
            },
            "search_after": bookmark,
            "sort": [
                {"date": "asc"},
                {"power": "desc"}
            ]
        }



while len(result['hits']['hits']) < size:
    res =es.search(index=es_index, body= body1)
    for el in res['hits']['hits']:
        result['hits']['hits'].append( el )
    print(len(result['hits']['hits']), res['hits']['hits'][-1]['sort'])
    bookmark = [res['hits']['hits'][-1]['sort'][0], str(result['hits']['hits'][-1]['sort'][1])]
    if size - len(result['hits']['hits']) < 10000:
        body_size = size - len(result['hits']['hits'])
        body1 = {"size": body_size,
                 "query": {
                     "term": {
                         "kiosk": kiosk
                     }
                 },
                 "search_after": bookmark,
                 "sort": [
                     {"date": "asc"},
                     {"power": "desc"}
                 ]
                 }
    else:
        body1 = {"size": 10000,
                "query": {
                    "term" : {
                        "kiosk" : kiosk
                    }
                },
                "search_after": bookmark,
                "sort": [
                    {"date": "asc"},
                    {"power": "desc"}
                ]
            }

print(result['hits']['hits'][0])
docs = pd.DataFrame()
res_list = []
iter = 0
# iterate each Elasticsearch doc in list
print("ncreating objects from Elasticsearch data.")
for num, doc in enumerate(result['hits']['hits']):
    # save data once in a while
    if num % 5000000 == 0 and num > 0:
        iter += 1
        docs = pd.concat(res_list, axis=0)
        docs.to_parquet(os.path.join(ROOT_DIR, 'data', f"temp_data{iter}.parquet"))
        del res_list #probably doesn't do much
        del docs
        res_list=[]
        docs = pd.DataFrame()
    if num % 100000 == 0:
        #track progress, check if mem is available for next batch
        print(num, len(res_list))
    # get _source data dict from document
    source_data = doc["_source"]
    # get _id from document
    _id = doc["_id"]
    # create a Series object from doc dict object
    doc_data = pd.Series(source_data, name=_id).to_frame().T
    res_list.append(doc_data)

docs = pd.concat(res_list, axis=0)

print(docs.info())
docs.to_parquet(os.path.join(ROOT_DIR, 'data', f'temp_data{iter+1}.parquet'))

print("nntime elapsed:", time.time() - start_time)