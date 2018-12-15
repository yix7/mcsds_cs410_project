import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk


def keep_useful_data(input_file):
    df = pd.read_csv(input_file, dtype={'postal_code': np.string_, 'review_from_user': np.int32})
    columns_to_be_removed = list(df.columns.values)
    columns_to_be_kept = ['review_from_user', 'text', 'business_and_id', 'user_and_id', 'adjusted_business_review_rating']
    for i in columns_to_be_kept:
        columns_to_be_removed.remove(i)
    df = df.drop(columns_to_be_removed, axis=1)
    return df


def training(df):
    training_data = df[(df['review_from_user'] == 1) | (df['review_from_user'] == 5)]
    testing_data = df[(df['review_from_user'] == 2) | (df['review_from_user'] == 3) | (df['review_from_user'] == 4)]
    training_data['sentiment'] = np.where(training_data['review_from_user'] == 5, 1, 0) # set to 1 if the rating is 5, set to 0 if the rating is 1
    X_train, X_test, y_train, y_test = train_test_split(training_data['text'], training_data['sentiment'], random_state=1234)
    vect = TfidfVectorizer().fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    model = LogisticRegression()
    model.fit(X_train_vectorized, y_train)
    predictions = model.predict(vect.transform(X_test))
    print('AUC score: ', roc_auc_score(y_test, predictions))
    precision, recall, f1, support = precision_recall_fscore_support(y_test, predictions, average='macro')
    print('Macro precision: {}, recall: {}, f1: {}'.format(precision, recall, f1))
    predictions_for_testing_data = model.predict(vect.transform(testing_data['text']))
    testing_data_sentiment = np.asarray(predictions_for_testing_data, dtype=np.int32)
    testing_data['sentiment'] = testing_data_sentiment
    all_review = pd.concat([training_data, testing_data])
    pos_review = all_review[all_review['sentiment'] == 1]
    pos_review_with_four_stars = pos_review[pos_review.adjusted_business_review_rating > 4]
    all_pos_reviews = ' '.join(pos_review['text'])
    all_pos_reviews_four_star = ' '.join(pos_review_with_four_stars['text'])
    return all_pos_reviews, all_pos_reviews_four_star


def find_common_unigram(all_pos_reviews, output_file):
    wordnet_lemmatizer = WordNetLemmatizer()
    # p_stemmer = PorterStemmer()
    stop_word_orig = stopwords.words('english')
    tokenizer = RegexpTokenizer(r'\w+')
    raw = all_pos_reviews.lower()
    tokens = tokenizer.tokenize(raw)
    stopped_tokens = []
    for i in tokens:
        if i not in stop_word_orig:
            stopped_tokens.append(i)
    lemmatized_tokens = [wordnet_lemmatizer.lemmatize(i) for i in stopped_tokens]
    fdist1 = nltk.FreqDist(lemmatized_tokens)
    with open(output_file, 'w') as fo:
        fo.write('word,count\n')
        for i,k in fdist1.most_common(1000):
            fo.write(str(i)+","+str(k)+"\n")


def find_common_bigram_trigram(words_for_removal, all_pos_reviews, all_pos_reviews_four_star):
    wordnet_lemmatizer = WordNetLemmatizer()
    stop_word_orig = stopwords.words('english')
    with open(words_for_removal, 'r') as fi:
        for i in fi:
            stop_word_orig.append(i.lower().strip())
    stop_words = set(stop_word_orig)
    tokenizer = RegexpTokenizer(r'\w+')
    raw = all_pos_reviews.lower()
    raw_four_star = all_pos_reviews_four_star.lower()
    tokens_all = tokenizer.tokenize(raw)
    tokens_four_star = tokenizer.tokenize(raw_four_star)
    stopped_tokens = [i for i in tokens_all if i not in stop_words]
    stopped_tokens_four_star = [i for i in tokens_four_star if i not in stop_words]
    lemmatized_tokens_all = [wordnet_lemmatizer.lemmatize(i) for i in stopped_tokens]
    lemmatized_tokens_all_four_star = [wordnet_lemmatizer.lemmatize(i) for i in stopped_tokens_four_star]
    # get common bigram, trigram for all pos reviews.
    bgs = nltk.bigrams(lemmatized_tokens_all)
    fdist2 = nltk.FreqDist(bgs)
    with open(r'..\output\all_pos_reviews_most_common_bigrams.csv', 'w') as fo:
        fo.write('word,count\n')
        for i, k in fdist2.most_common(300):
            fo.write(' '.join(i))
            fo.write(',' + str(k) + "\n")
    tgs = nltk.trigrams(lemmatized_tokens_all)
    fdist3 = nltk.FreqDist(tgs)
    with open(r'..\output\all_pos_reviews_most_common_trigrams.csv', 'w') as fo:
        fo.write('word,count\n')
        for i, k in fdist3.most_common(300):
            fo.write(' '.join(i))
            fo.write(',' + str(k) + "\n")
    # get common bigram, trigram for all 4 star reviews.
    bgs = nltk.bigrams(lemmatized_tokens_all_four_star)
    fdist2 = nltk.FreqDist(bgs)
    with open(r'..\output\4star_pos_reviews_most_common_bigrams.csv', 'w') as fo:
        fo.write('word,count\n')
        for i, k in fdist2.most_common(300):
            fo.write(' '.join(i))
            fo.write(',' + str(k) + "\n")
    tgs = nltk.trigrams(lemmatized_tokens_all_four_star)
    fdist3 = nltk.FreqDist(tgs)
    with open(r'..\output\4star_pos_reviews_most_common_trigrams.csv', 'w') as fo:
        fo.write('word,count\n')
        for i, k in fdist3.most_common(300):
            fo.write(' '.join(i))
            fo.write(',' + str(k) + "\n")


df = keep_useful_data(r'..\output\adjusted_review_result.csv')
all_pos_reviews, all_pos_reviews_four_star = training(df)
find_common_unigram(all_pos_reviews, r'common_words_unigram.csv')
find_common_bigram_trigram(r'common_words_for_removal.csv', all_pos_reviews, all_pos_reviews_four_star)