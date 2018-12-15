import pandas as pd
import numpy as np


'''
    Use bm25 concept to adjust weight based on count.
'''


def bm25_adj(count, k=0.15):
    return count*(k+1)/(count+k)


'''
    Step 1:
    If a user has reviewed more Chinese food, rating weight increases based on review counts (max up to 0.7 extra)
    Return an updated dataframe to include the adjusted weight.
'''


def user_food_review_count(file):
    file = pd.read_csv(file, index_col=None, header=0)
    user_review_count_series = file['user_and_id'].value_counts(sort=True, ascending=True)
    adj_on_user_review = user_review_count_series.apply(bm25_adj, args=(0.7,))
    df = adj_on_user_review.to_frame(name='adj_based_on_food_review_count')
    df.reset_index(level=0, inplace=True)
    df = df.rename(index=str, columns={'index':'user_and_id'})
    results = file.merge(df, on='user_and_id')
    return results


'''
    Step 2:
    If a user has more total review count, (regardless of what food, service etc) is high, rating weight increases based on total review count (max up to 0.2 extra)
    Return an updated dataframe to include the adjusted weight.
'''


def user_total_review_count(df):
    df['adj_based_on_total_review_count'] = df['user_total_review_count'].apply(bm25_adj, args=(0.2,))
    return df


'''
    Step 3:
    If a user has ever been an elite member, based on the number of years of him/her being an elite member, increases the rating weight accordingly. (max up to 0.4 extra) 
    If a user has never been an elite member, set weight adjustment as 0.95.
'''


def adj_based_on_elite_member(elite_status, k=0.1):
    if elite_status == 'None':
        return 0.95
    else:
        count = len(elite_status.split(','))
        return bm25_adj(count, k=k)


def elite_member_count(df):
    df['adj_based_on_elite_status'] = df['elite'].apply(adj_based_on_elite_member, args=(0.4,))
    return df


'''
    Step 4:
    Check the useful count of that review, increase rating weight accordingly (max up to 0.9)
    If a review has 0 useful count, set weight adjustment as 0.9.
'''


def adj_based_on_usefulness_count(count, k=0.2):
    if count == 0:
        return 0.9
    else:
        return bm25_adj(count, k=k)


def review_usefulness_count(df):
    df['adj_based_on_useful_count'] = df['useful'].apply(adj_based_on_usefulness_count, args=(0.9,))
    return df


'''
    Step 5:
    Compare the review rating to average rating of that user, adjust rating weight accordingly (max up to 0.4). 
    Increase the weight if this particular review rating is more deviated compared to average rating from the user.
'''


def adj_based_on_avg_rating(rating_delta, k=0.2):
    return 1+k/(5-1)*rating_delta


def average_rating(df):
    df['adj_based_on_avg_rating'] = df['review_from_user'].sub(df['average_stars'], fill_value=0).abs().apply(adj_based_on_avg_rating, args=(0.4,))
    return df


'''
    Step 6:
    Check if each review is based on food topic or not. If it's food related topic, increase the weight accordingly (max up to 1.0)
    All the topics are reviewed by the user and determine whether they are food related or not.
'''


def adj_based_on_topic(topic, topic_file=r'../output/mined_topics.csv'):
    food_topic = dict()
    count = -1
    with open(topic_file, 'r') as fi:
        for i in fi:
            count += 1
            if count == 0:
                continue
            topic_list = i.split(',')
            food_topic[int(topic_list[0])] = int(topic_list[2])
            # print(topic_list[0], topic_list[2])
    if food_topic[int(topic)] == 1:
        return 2
    else:
        return 1


def review_as_food_topic(df):
    df['adj_based_on_topic'] = df['topic'].apply(adj_based_on_topic)
    return df


'''
    Adjust review ratings based on above steps
    Adjust review weight based on above steps
'''


def adj_review_rating(df, *argv):
    df['adjusted_review_rating'] = df['review_from_user']

    for arg in argv:
        df['adjusted_review_rating'] = df['adjusted_review_rating'].mul(arg)
    return df


def adj_review_weight(df, str, *argv):
    df[str] = 5.0
    for arg in argv:
        df[str] = df[str].mul(arg)
    return df


'''
    Apply all above steps.
'''


def apply_review_weight_adjustment(input_datalog_file):
    result = user_food_review_count(input_datalog_file)
    result = user_total_review_count(result)
    result = elite_member_count(result)
    result = review_usefulness_count(result)
    result = average_rating(result)
    result = review_as_food_topic(result)
    result = adj_review_weight(result, 'adjusted_review_weight', result['adj_based_on_food_review_count'],
                          result['adj_based_on_total_review_count'], result['adj_based_on_elite_status'],
                          result['adj_based_on_useful_count'], result['adj_based_on_avg_rating'], result['adj_based_on_topic'])
    result = adj_review_rating(result, result['adj_based_on_food_review_count'], result['adj_based_on_total_review_count'],
                               result['adj_based_on_elite_status'], result['adj_based_on_useful_count'], result['adj_based_on_avg_rating'], result['adj_based_on_topic'])
    # Calculate original business rating
    exact_review_from_user = result.groupby('business_and_id')['review_from_user'].agg(np.mean).to_frame().rename(index=str,
                                                                                                             columns={
                                                                                                                 'review_from_user': 'exact_review_from_user'})
    result = result.merge(exact_review_from_user, on='business_and_id')
    # Calculate adjusted business rating
    business_review_sum = result.groupby('business_and_id')['adjusted_review_rating'].agg(np.sum).to_frame().rename(
        index=str, columns={'adjusted_review_rating': 'sum_adj_review_rating'})
    business_review_weight = result.groupby('business_and_id')['adjusted_review_weight'].agg(np.sum).to_frame().rename(
        index=str, columns={'adjusted_review_weight': 'sum_adj_review_weight'})
    merge_business_review = pd.merge(business_review_sum, business_review_weight, on='business_and_id')
    merge_business_review['adjusted_business_review_rating'] = 5 * merge_business_review['sum_adj_review_rating'].div(
        merge_business_review['sum_adj_review_weight'])
    result = result.merge(merge_business_review, on='business_and_id')
    return result

df = apply_review_weight_adjustment(r'../output/merged_with_topic_listing.csv')
df.to_csv(r'../output/adjusted_review_result.csv', encoding='utf-8', index=False)