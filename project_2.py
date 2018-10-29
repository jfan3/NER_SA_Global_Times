# -*- coding: utf-8 -*-

import multiprocessing as mp
import time
from urllib.request import urlopen, urljoin
from bs4 import BeautifulSoup
import re
from stanfordcorenlp import StanfordCoreNLP
import pickle



def crawl(url):
    '''This function reals the html string from url'''
    response = urlopen(url)
    time.sleep(0.1)             # slightly delay for downloading
    return response.read().decode('UTF-8')


def get_links_chinese(url):
    '''This function returns the links to articles from an index link (for Chinese)'''
    html = crawl(url)
    soup = BeautifulSoup(html,'lxml')
    links_news=list(set([i['href'] for i in soup.find_all('a', {"href": re.compile('http://opinion.huanqiu.com/hqpl/*')})[:-12]]))
    return links_news

def get_links_english(url):
    '''This function returns the links to articles from an index link (for Chinese)'''
    html = crawl(url)
    soup = BeautifulSoup(html,'lxml')
    links_news=list(set([i['href'] for i in soup.find_all('a', {"href": re.compile('http://www.globaltimes.cn/content/')})]))
    return links_news

def parse_ch(link_news):
    '''This function parses the news html string into the title and body that we care abuot (for Chinese)'''
    soup2=BeautifulSoup(crawl(link_news),'lxml')
    bodies=[i.text for i in soup2.find_all('p')]
    body = "".join(bodies)[:-89]
    body = re.sub('\u3000', '', body)
    # print(body)
    body = re.sub('\r\n', '', body)
    title = soup2.h1.text
    return title, body

def parse_en(link_news):
    '''This function parses the news html string into the title and body that we care abuot (for English)'''
    soup=BeautifulSoup(crawl(link_news),'lxml')
    if soup.p is None:
        body = soup.find_all('div',{'class':'span12 row-content'})
        body = [i.text for i in body]
        body = "".join(body)
    else:
        body=soup.p.text
    title=soup.h3.text
    return title, body

# parse_english(url)
def construct_corpus_ch(urls, q_ch, ch_dict):
    '''This function uses a queue and dictionary object to contruct the corpus of all crawled articles'''
    for url in urls:
        #         print(url)
        links_news = get_links_chinese(url)
        for link in links_news:
            try:
                title, body = parse_ch(link)
                # body = json.loads(body)
                ch_dict[title] = body
                # print(title)
                q_ch.put(title)
            except TimeoutError:
                continue



def construct_corpus_en(urls, q_en, en_dict):
    '''This function uses a queue and dictionary object to contruct the corpus of all crawled articles'''
    for url in urls:
        links_news = get_links_english(url)
        for link in links_news:
            try:
                title, body = parse_en(link)
                en_dict[title]=body
                q_en.put(title)
            except TimeoutError:
                continue

def process(q, corpi_dict, ners_dict, nlp):
    '''This function converts the corpus of articles into NER objects, and store it in a dictionary.'''
    while (not q.empty()):
        temp_corpus = str(corpi_dict[q.get()])
        temp_corpus=temp_corpus.replace("'", '"')
        temp_corpus = re.sub("u'","\"",temp_corpus)

        temp_ner = nlp.ner(temp_corpus)
        for word, typ in temp_ner:
            if typ not in list(ners_dict.keys()):
                ners_dict[typ]=[]
            temp = ners_dict[typ]
            temp.append(word)
            ners_dict[typ]= temp

if __name__ == '__main__':
    base_url_chinese = "http://opinion.huanqiu.com/hqpl"
    base_url_english = "http://www.globaltimes.cn/opinion/editorial/index.html"

    numbers = range(2, 31)
    chinese_to_crawl = [base_url_chinese]
    supp = ["http://opinion.huanqiu.com/hqpl/" + str(i) + ".html" for i in numbers]
    chinese_to_crawl.extend(supp)
    chinese_to_crawl
    english_to_crawl = [base_url_english]
    supp = ["http://www.globaltimes.cn/opinion/editorial/index" + str(i) + ".html" for i in numbers]
    english_to_crawl.extend(supp)


    en_dict = mp.Manager().dict()
    ch_dict = mp.Manager().dict()
    q_ch = mp.Queue()
    q_en = mp.Queue()
    p_en = mp.Process(name="en_producer", target=construct_corpus_en, args=(english_to_crawl, q_en, en_dict,))
    p_ch = mp.Process(name="ch_producer", target=construct_corpus_ch, args=(chinese_to_crawl, q_ch, ch_dict,))


    p_en.start()
    p_ch.start()
    print(f"Producer processes {p_ch.pid,p_en.pid} started...")

    p_en.join()
    p_ch.join()
    print("Producer processes ended...")


    with open('en_dict.pickle', 'wb') as output_en:
        pickle.dump(en_dict, output_en, protocol=pickle.HIGHEST_PROTOCOL)
    with open('ch_dict.pickle', 'wb') as output_ch:
        pickle.dump(ch_dict, output_ch, protocol=pickle.HIGHEST_PROTOCOL)


    address = 'http://ec2-18-222-166-129.us-east-2.compute.amazonaws.com'
    nlp_ch = StanfordCoreNLP(address, port=9000, lang='zh')
    nlp_en = StanfordCoreNLP(address, port=9000, lang='en')

    ners_dict_en = mp.Manager().dict()
    ners_dict_ch = mp.Manager().dict()

    print("Starting consumer processes....")

    consumer_en = mp.Process(target=process, args=(q_en, en_dict, ners_dict_en, nlp_en,))
    consumer_ch = mp.Process(target=process, args=(q_ch, ch_dict, ners_dict_ch, nlp_ch,))
    consumer_ch.start()
    consumer_en.start()

    consumer_en.join()
    consumer_ch.join()
    print("Consumer processes end...")
    ners_dict_en = dict(ners_dict_en)
    ners_dict_ch = dict(ners_dict_ch)
    print("Writing NER results to pickles for further analyses")
    with open('ner_en.pickle', 'wb') as output_en:
        pickle.dump(ners_dict_en, output_en, protocol=pickle.HIGHEST_PROTOCOL)
    with open('ner_ch.pickle', 'wb') as output_ch:
        pickle.dump(ners_dict_ch, output_ch, protocol=pickle.HIGHEST_PROTOCOL)

