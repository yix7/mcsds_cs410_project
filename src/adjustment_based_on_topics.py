import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models


# setup pre-processing on the reviews
wordnet_lemmatizer = WordNetLemmatizer()
p_stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+')


def extract_topics(input_file):
    df = pd.read_csv(input_file, index_col=None, header=0)
    doc_set = df['text'].dropna().tolist()[:]
    texts = []
    for doc in doc_set:
        # convert all the reviews to lower case
        raw = doc.lower()
        # tokenize the review
        tokens = tokenizer.tokenize(raw)
        # remove stop words
        stopped_tokens = [i for i in tokens if not i in stop_words]
        # lemmatize the words
        lemmatized_tokens = [wordnet_lemmatizer.lemmatize(i) for i in stopped_tokens]
        # word stemming
        stemmed_tokens = [p_stemmer.stem(i) for i in lemmatized_tokens]
        # add all the processed review to a list
        texts.append(stemmed_tokens)
        # if count % 100 == 0:
        #     print(count)
        # count += 1
    # create a dictionary
    dictionary = corpora.Dictionary(texts)
    # create a corpus
    corpus = [dictionary.doc2bow(text) for text in texts]
    # apply tfidf weighting
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    # create ldamodel based on the corpus, extract 10 topics
    ldamodel = models.ldamodel.LdaModel(corpus_tfidf, num_topics=6, id2word=dictionary, passes=1)
    # print out all the extracted topics
    with open('..\output\mined_topics.csv', 'w') as fo:
        fo.write("TopicNumber,TopicContent,FoodRelated - Enter 1 if yes enter 0 otherwise\n")
        for i in range(6):
            fo.write("{0},{1}\n".format(i, ldamodel.print_topic(i, 5)))
    lda_corpus = ldamodel[corpus]
    topic_for_each_doc = list()
    for doc in lda_corpus:
        max_score = 0
        matching_topic = -1
        for topic in doc:
            if topic[1] > max_score:
                max_score = topic[1]
                matching_topic = topic[0]
        topic_for_each_doc.append(matching_topic)
    df['topic'] = topic_for_each_doc
    return df


df = extract_topics(r'../output/merged_user_revivew_business.csv')
df.to_csv(r'../output/merged_with_topic_listing.csv', encoding='utf-8', index=False)