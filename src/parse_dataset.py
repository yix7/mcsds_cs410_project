import pandas as pd
import glob
from fsplit.filesplit import FileSplit
import os, gc


'''
    Columns to be removed from the original yelp dataset.
'''

business_labels_drop = ['hours', 'is_open', 'latitude', 'longitude', 'neighborhood', 'address', 'attributes', 'categories', 'city', 'postal_code', 'state']
review_labels_drop = ['cool', 'funny']
user_labels_drop = ['friends', 'useful', 'funny', 'cool', 'fans', 'compliment_hot', 'compliment_more', 'compliment_profile',
                    'compliment_cute', 'compliment_list', 'compliment_note', 'compliment_plain', 'compliment_cool',
                    'compliment_funny', 'compliment_writer', 'compliment_photos']
merge_labels_drop = ['business_id', 'review_id', 'user_id']

# list all the dataset names that'll be used.
business_dataset_name = '../yelp_dataset/yelp_academic_dataset_business.json'
review_dataset_name = '../yelp_dataset/yelp_academic_dataset_review.json'
user_dataset_name = '../yelp_dataset/yelp_academic_dataset_user.json'


def parse_business_dataset(dataset_name):

    '''
        Parse the business json log from yelp dataset.
        Filter in the businesses fit below criteria:
            1. It is a restaurant.
            2. Is is a Chinese restaurant. (we use Chinese restaurant to prove the concept, tool can be expanded to other kinds of food as well.
            3. The review count of this business is at least 100. This is to avoid data being skewed due to limited reviews.
            4. The business is currently open.
    '''

    # Columns which are no use for this project will be removed from the original yelp dataset.
    # Read business dataset and filter in only the Chinese restaurants which has >= 100 review counts,
    # is still open, and have average review of >= 3.5.
    business_df = pd.read_json(dataset_name, lines=True)
    business_df = business_df.dropna()
    business_df = business_df[business_df['categories'].str.contains('Restaurants')]
    business_df = business_df[business_df['categories'].str.contains('Chinese')]
    business_df = business_df[
        (business_df.review_count >= 100) & (business_df.is_open == 1) & (business_df.stars >= 3.5)]
    business_df.drop(business_labels_drop, axis=1, inplace=True)
    business_df = business_df.rename(index=str, columns={'stars': 'business_avg_star', 'name': 'business_name',
                                                         'review_count': 'business_review_count'})
    return business_df


def parse_review_dataset(dataset_name, business_df):

    '''
        Parse the review json log from yelp dataset.
        Due to the size of the review json file, split it into 6 smaller files, and parse each file separately, in order to process them.
        Filter in only the reviews belong to above businesses.
        Merge filtered individual files together.
    '''

    # since the review json file size is too big (>4GB), split review json file to smaller files for post-processing
    fs = FileSplit(
        file=dataset_name,
        splitsize=800000000,
        output_dir=os.path.dirname(dataset_name))
    fs.split(include_header=True)

    # Get the name of files splitted
    subfile_name = dataset_name.replace('.json', '_')
    subfiles = glob.glob(subfile_name+'*')

    # Read all the subfiles and combine them into one dataframe for processing
    list_ = []
    for file_ in subfiles:
        df_ = pd.read_json(file_, lines=True)
        df_ = df_.dropna()
        df_.drop(review_labels_drop, axis=1, inplace=True)
        df_ = df_[df_['business_id'].isin(list(set(business_df['business_id'].tolist())))]
        list_.append(df_)
    all_review_df = pd.concat(list_)
    all_review_df = all_review_df.rename(index=str, columns={'stars':'review_from_user'})
    return all_review_df


def parse_user_dataset(dataset_name, all_review_df):

    '''
        Parse the user json log fom yelp dataset.
        Filter in only the users belong to above reviews.
        After filtering the data, convert to csv file.
    '''

    user_df = pd.read_json(dataset_name, lines=True)
    user_df = user_df.dropna()
    user_df.drop(user_labels_drop, axis=1, inplace=True)
    user_df = user_df[user_df['user_id'].isin(list(set(all_review_df['user_id'].tolist())))]
    user_df = user_df.rename(index=str, columns={'name':'user_name', 'review_count':'user_total_review_count'})
    return user_df


def parse_dataset():

    '''
        Merge business, review, user three files together.
        Save the merged file to csv.
    '''

    # parse all the dataset
    business_df = parse_business_dataset(business_dataset_name)
    all_review_df = parse_review_dataset(review_dataset_name, business_df)
    # combine all the dataset
    results = all_review_df.merge(business_df, on='business_id')
    del business_df
    # del all_review_df
    gc.collect()
    user_df = parse_user_dataset(user_dataset_name, all_review_df)
    results = results.merge(user_df, on='user_id')
    del user_df
    del all_review_df
    gc.collect()
    results['business_and_id'] = results['business_name'].str.cat(results['business_id'], sep='_')
    results['user_and_id'] = results['user_name'].str.cat(results['user_id'], sep='_')
    results.drop(merge_labels_drop, axis=1, inplace=True)
    return results
    # results.to_csv(output_file_name, encoding='utf-8', index=False)


output_file_name = r'../output/merged_user_revivew_business.csv'
df = parse_dataset()
df.to_csv(output_file_name, encoding='utf-8', index=False)