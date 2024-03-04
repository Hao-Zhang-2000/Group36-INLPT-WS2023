import requests
import json
import xml.etree.ElementTree as ET
import time
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool, Manager




def get_abstract(id, failure_time=0):
    base_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
    url = f'{base_url}?db=pubmed&id={id}&retmode=xml'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    try:
        response = requests.get(url,headers=headers)

        data = ET.fromstring(response.text)
        abstract_texts = [''.join(abstract_text.itertext()).strip() for abstract_text in
                          data.findall('.//AbstractText')]
        # title_elements = data.findall('.//Title')
        if abstract_texts:
            abstract = '\n'.join(abstract_texts)
            lastname = data.findall('.//LastName')
            forename = data.findall('.//ForeName')
            author = [[lastname[i].text + ' ' + forename[i].text] for i in range(len(lastname))]
            #print(author)
            title = data.findall('.//ArticleTitle')[0].text
            #articleid = [[i.text] for i in data.findall('.//ArticleId')]
            #articleid = [[i.text] for i in data.findall('.//ArticleId') if i.find('ReferenceList') is None]

            article_ids_in_articleidlist = data.findall('.//ArticleId')

            # Find all ArticleId tags in ReferenceList
            article_ids_in_referencelist = data.find('.//ReferenceList')
            if(article_ids_in_referencelist):
                filter = article_ids_in_referencelist.findall('.//ArticleId')
                articleid = [
                    [article_id.text] for article_id in article_ids_in_articleidlist
                    if article_id not in filter
                ]
            else:
                articleid = [
                    [article_id.text] for article_id in article_ids_in_articleidlist
                ]

            return abstract,author,url,title,articleid
        else:
            #print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n")
            return None,None,None,None,None
    except:

        #print(f'###{url}### : retrival failed : {failure_time + 1}\n', flush=True)
        if (failure_time < 2):  # maximum try of 3 times
            time.sleep(1.0)
            return get_abstract(id, failure_time + 1)
        else:
            #print(f'###{id}###: failed 3 times, abandoning \n')
            return None,None,None,None,None

        # return None


def get_abstract_multiproc(proc_num, total_process, lock, lists, lists_author, lists_url, lists_title, _article_ids, lists_failed,lists_aid):
    #proc_num, total_process, lock, lists, _article_ids = args_tuple
    #print("process" + str(proc_num) + "\n")
    article_num = len(_article_ids)
    current_proc_num = proc_num

    while (current_proc_num < article_num):
        #print(current_proc_num)
        abstract,author,url,title,aid = get_abstract(_article_ids[current_proc_num])
        if(abstract != None):
            print(f'###Thread:{current_proc_num} complete!###', flush=True)
            with lock:
                lists.append(abstract)
                lists_author.append(author)
                lists_url.append(url)
                lists_title.append(title)
                lists_aid.append(aid)
        else:
            print(f'###Thread:{current_proc_num} failed! id:{_article_ids[current_proc_num]}###', flush=True)
            lists_failed.append(_article_ids[current_proc_num])
        current_proc_num += total_process
        time.sleep(0.5)



def get_abstract_multiproctest(proc_num):
    #proc_num, total_process, lock, lists, _article_ids = args_tuple
    print("process" + proc_num + "\n")

# init_dt()
# On Windows the subprocesses will import (i.e. execute) the main module at start.
# You need to insert an if __name__ == '__main__':
#    guard in the main module to avoid creating subprocesses recursively.
if __name__ == '__main__':

    retstart = 0
    len_last_response = 10_000

    how_many = 10
    # p = Pool(processes=how_many)

    # func = partial(get_abstract_multiproc, how_many)
    # zip(ind, itertools.repeat(how_many))
    # data = p.map(get_abstract_multiproc, ind)

    pool = multiprocessing.Pool(how_many)
    manager = Manager()
    l = manager.Lock()
    my_list = manager.list([])
    my_list_url = manager.list([])
    my_list_title = manager.list([])
    my_list_author = manager.list([])
    article_ids = manager.list([])
    my_list_failed = manager.list([])
    my_list_aid = manager.list([])
    # article_ids = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    len_last_response, retstart, article_ids

    for i in range(11): #2013 - 2023
        print(f"#####################yeah:{2013+i}######################\n")
        response = requests.get(
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=intelligence&retmode=json&retmax=10000&mindate={2013+i}/01/01&maxdate={2013+i}/12/31",headers=headers)
        data = response.json()

        try:
            id_list = data["esearchresult"]["idlist"]
        except:
            print("xx\n")
            time.sleep(0.2)
            continue
        len_response = len(id_list)
        retstart += len_response
        len_last_response = len_response

        article_ids.extend(id_list)

        print(f'###article numbers:{len(article_ids)}###\n')
        ind = [(i, how_many, l, my_list,my_list_author, my_list_url, my_list_title, article_ids, my_list_failed,my_list_aid) for i in range(how_many)]
        pool.starmap(get_abstract_multiproc, ind)


        len(my_list)
        valid_abstracts = []
        valid_author = []
        valid_urls = []
        valid_title = []
        valid_aid = []
        id = 0
        for abstract in my_list:
            if "intelligence" in abstract.lower():
                valid_abstracts.append(abstract)
                valid_author.append(my_list_author[id])
                valid_urls.append(my_list_url[id])
                valid_title.append(my_list_title[id])
                valid_aid.append(my_list_aid[id])
            id += 1
        with open(f"valid_abstracts{2013+i}.json", "w+") as f:
            f.write(json.dumps({
                "abstracts": valid_abstracts,
                "authior": valid_author,
                "url": valid_urls,
                "title": valid_title,
                "article_id": valid_aid
            }))
        with open(f"failed{2013+i}.json", "w+") as f:
            f.write(json.dumps({
                "failed_id": list(my_list_failed),
            }))
        my_list_failed[:] = []
        article_ids[:] = []
        my_list[:] = []
        my_list_url[:] = []
        my_list_title[:] = []
        my_list_author[:] = []
        my_list_aid[:] = []

    pool.close()
    pool.join()
