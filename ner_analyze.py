from wordcloud import WordCloud
from collections import *
import pickle
import multiprocessing as mp
import os

def analysis (ners_dict, specifier, lan='en', font="G:\My Drive\Fall 2018\CAPP 30525\week3\simfang.ttf"):
    '''This function performs wordcloud analysis on the NER restuls.
    Specifier can be ['DATE', 'O', 'COUNTRY', 'TITLE', 'PERSON', 'ORGANIZATION', 'NUMBER', 'LOCATION', 'TIME', 'DEMONYM',
        'MONEY', 'STATE_OR_PROVINCE', 'GPE', 'MISC', 'ORDINAL', 'CITY', 'IDEOLOGY', 'PERCENT', 'CAUSE_OF_DEATH', 'FACILITY',
        'CRIMINAL_CHARGE', 'RELIGION', 'NATIONALITY']'''

    try:
        temp = Counter(ners_dict[specifier])
    except KeyError:
        print("Key incorrect!")
    ordered = temp.most_common()
    with open (f"output\{lan}_{specifier}.txt",'w', encoding="utf-8") as f:
        f.write(f"15 most common mentions in category {specifier} are:\n")
        for word, count in ordered[:15]:

            f.write(f"{word} with {count} mentions;\n")

    # create word cloud
    if lan == 'zh':
        wordcloud = WordCloud(collocations=False, font_path=font, width=5000, height=3000, margin=2)
    else:
        wordcloud = WordCloud(width=5000, height=3000, margin=2)

    wordcloud.generate_from_frequencies(frequencies=temp)
    # plt.figure()
    # plt.imshow(wordcloud, interpolation="bilinear")
    # plt.axis("off")
    # plt.show()
    # plt.close()
    wordcloud.to_file(f'output\{lan}_{specifier}.png')


if __name__ == '__main__':
    with open('ner_en.pickle', 'rb') as handle:
        ner_en = pickle.load(handle)
    with open('ner_ch.pickle', 'rb') as handle:
        ner_ch = pickle.load(handle)

    specifiers = ['COUNTRY', 'TITLE', 'PERSON', 'ORGANIZATION', 'LOCATION', 'STATE_OR_PROVINCE', 'CITY', 'IDEOLOGY',
                  'RELIGION']
    # with mp.Pool(os.cpu_count()) as p:
    #     p.apply(analysis, args=(ner_ch,specifiers,'zh'))
    #     p.apply(analysis, args=(ner_en, specifiers))
    for specifier in specifiers:
        analysis(ner_ch,specifier,'zh')
        analysis(ner_en,specifier)
