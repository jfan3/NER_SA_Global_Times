import matplotlib.pyplot as plt
import os
from stanfordcorenlp import StanfordCoreNLP
from collections import *
from itertools import *
import json
import numpy as np
import pickle
from functools import *

def find_relevant_art(pattern, corpus):
    '''This function finds the relevant articles in corpus based on a keyword'''
    temp = dict()
    bodies = []
    for title,body in corpus.items():
        if pattern in body:
            temp[title]=body
            bodies.append(body)
    return dict(list(temp.items()))

def sent_analysis(text,nlp,props={'annotators': 'sentiment',
             'pipelineLanguage': 'en',
             'outputFormat': 'json'}):
    '''This function performs sentiment analysis on text, and return a list of sentiment values for each sentence in the text'''
    total = np.empty(0)
    results = json.loads(nlp.annotate(text, properties=props))
    n = len(results['sentences'])
    sen_vals = np.empty(n)
    for i, sentence in enumerate(results['sentences']):
        sen_vals[i] = sentence['sentimentValue']
    total = np.append(total, sen_vals)
    return total

def get_results(pattern, corpus, nlp, props={'annotators': 'sentiment',
             'pipelineLanguage': 'en',
             'outputFormat': 'json'}):
    '''This function combines find_relevant_art and sent_analysis'''
    temp = find_relevant_art(pattern, corpus)
    all = np.empty(0)

    for title, body in temp.items():
        try:
            snippet_scores = sent_analysis(body,nlp,props=props)
            all = np.append(all, snippet_scores)
        except json.decoder.JSONDecodeError:
            # print(body)
            corpus['title']=""
            continue
    return all.mean(), all.std()

def gen(en_dict, nlp,patterns):
    '''This function applies get_results to a list of patterns'''
    results = []
    for pattern in patterns:
        result = get_results(pattern, en_dict, nlp)
        results.append(result)
    return results
def plotting(results, patterns, category):
    '''This function plots the sentiment analysis results'''
    avgs = np.empty(0)
    stds = np.empty(0)
    for avg,std in results:
        avgs = np.append(avgs, avg)
        stds = np.append(stds, std)
    x = np.arange(len(patterns))
    fig, ax = plt.subplots()
    ax.bar(x, avgs, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Sentiment Score (1:negative, 2:neurtral, 3:positive)')
    ax.set_xticks(x)
    ax.set_xticklabels(patterns)
    ax.set_title(f'Sentiment Analysis of Global Times Opinion Articles for {category}')
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    plt.savefig(f'output\senti_{category}.png')
    # plt.show()
if __name__ == '__main__':
    with open('en_dict.pickle', 'rb') as handle:
        en_dict = pickle.load(handle)
    address = 'http://ec2-18-216-224-118.us-east-2.compute.amazonaws.com'
    nlp = StanfordCoreNLP(address, port=9000, lang='en')
    countries = ['China','US','Russia']
    ideologies = ['socialist', 'modernization', 'independence', 'democracy']
    organizations = ['CPC', 'WTO', 'White House']
    persons = ['Trump', 'Xi']
    country_results = gen( en_dict, nlp, countries)
    plotting(country_results, countries, 'Countries')
    ideologies_results = gen(en_dict,nlp,ideologies)
    plotting(ideologies_results,ideologies,'Ideologies')
    organizations_results = gen(en_dict,nlp,organizations)
    plotting(organizations_results,organizations,'Organization')
    persons_results = gen(en_dict,nlp,persons)
    plotting(persons_results,persons,'Organization')


    # func =  partial(gen, en_dict,nlp )
    # with mp.Pool(8) as pool:
    #     results = pool.map(func,countries)
    # # plotting(results,organizations,'Organizations')
    # plotting(results,countries,'Country_3')
    # plotting(results,organizations,'organizations')