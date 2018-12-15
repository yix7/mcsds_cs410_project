# Authentic Cuisine Detector


### Introduction

An analytic system was built to help users find the best food in town and the most favorable, authentic dishes among the top-notch restaurants. The system was built on the dataset provided by Yelp. We used Chinese food as an example in this project.


### Implementation Details

1) Download yelp dataset from https://www.yelp.com/dataset. Please download the entire dataset (>4G) and un-tar the file. Place these 3 json files under the **yelp_dataset** folder. They will be used for this project.
    * yelp_academic_dataset_business.json
    * yelp_academic_dataset_review.json
    * yelp_academic_dataset_user.json
2) Parse yelp dataset. This step is achieved with **src/parse_dataset.py**.
    * Load json files to python pandas dataframe for data processing.
    * Parse yelp business json file - _yelp_dataset/yelp_academic_dataset_business.json_, only filter in the business fits below criteria:
        - Business is a Chinese restaurant. (use Chinese food as the example here)
        - Total review count >= 100. (we want to avoid businesses with low review count, whose rating may be skewed)
        - Current review rating >= 3.5. (focus on the restaurants that have ok rating based on existing rating scheme)
    * Parse yelp review json file - _yelp_dataset/yelp_academic_dataset_review.json_
        - Due to the big size of the review file, split it into 6 sub-files to avoid hitting memory limitation.
        - Filter in the businesses that were identified in the previous step.
        - Get a list of users that were involved in rating these businesses, prepare for next step.
        - After processing all sub-files, merge them into one datalog.
    * Parse yelp user json file - _yelp_dataset/yelp_academic_dataset_user.json_
        - Filter in the users that were involved in rating the businesses identified, based on the list created from previous steps. 
    * Merge yelp business, review, user files into one datalog, for further processing.
    * Remove all the information from yelp dataset that are not needed for this project.
3) Check the topic each review covers, tag each one as either food related or non-food related. This step is achieved with **src/adjustment_based_on_topics.py**.
    * As an important part of the project, we use LDA to generate topics from the all the reviews. Generated 7 topics for illustration. - _output/mined_topics.csv_
    * Based on the topics generated, we manually tag them as either food related or non-food related - update the 3rd column: enter 1 if user thinks this topic is food related; enter 0 if user thinks this topic is not food related. Save this file.
4) Adjust business rating based on whether the user has a good knowledge of the cuisine and his/her authority, as well as whether the review is food related or not. Use a technique similar to TF for weight adjustment. This step is achieved with **src/review_adjustment.py**.
    * If a user has reviewed more Chinese food, rating weight increases based on review counts. (upto 70% weight increase)
    * If a user has more total review count, (regardless of the kind of business reviewed) is high, rating weight increases based on total review count. (upto 20% weight increase)
    * If a user has ever been an elite member, based on the number of years of him/her being an elite member, increases the rating weight accordingly. (upto 40% weight increase)
    * Check the “useful” count of that review, increase rating weight accordingly. (upto 90% weight increase)
    * Compare the review rating to average rating of that user, adjust rating weight accordingly. If the user has given a rating that’s more deviated compared to his/her average review rating, increase rating weight accordingly. (upto 40% weight increase)
    * Check if review is food related or not based on step 3, increase weight if the review is food related. (100% weight increase)
5) Find the best dishes from all the restaurants, and only the restaurants having >4 stars after adjustment. This step is achieved with **src/best_dishes.py**.
    * Mark all the 5-star reviews as positive sentiment. Mark all the 1-star reviews as negative sentiment.
    * Based on 5-star and 1-star reviews, train a model to identify the sentiment of a review.
    * Since it's not clear if a 2/3/4-star review is positive or negative, apply the model to 2/3/4-star reviews, and tag whether a review is positive or negative.
    * From all the positive reviews, get the unigram word distribution.
    * From the 1000 most frequent unigram words, manually tag whether each one is dish name related or not.
    * Filter out all the frequent words that are not dish name related from the reviews.
    * From all the filtered positive reviews, get the most frequent bi-gram and tri-gram words, and they are our best dishes.


### Installations and Dependencies
* Programming Language: Python 3.x
* Modules: pandas; numpy; glob; filesplit; nltk; gensim; sklearn.
```
    pip install pandas numpy glob filesplit nltk gensim sklearn
```
 
 
### How to Use
1. Download yelp dataset from https://www.yelp.com/dataset. Please download the entire dataset (>4G) and un-tar the file. Place these 3 json files under the **yelp_dataset** folder. They will be used for this project.
    * yelp_academic_dataset_business.json
    * yelp_academic_dataset_review.json
    * yelp_academic_dataset_user.json
2. Parse yelp dataset:
    ```
    python src/parse_dataset.py
    ```
3. Check topics all the reviews cover and the topic of each review:
    ```
    python src/adjustment_based_on_topics.py
    ```
4. Tag whether a topic is food related or not. Open output/mined_topics.csv and fill up the 3rd column. Enter 1 if user thinks this topic is food related, enter 0 if user thinks this topic is not food related.
5. Adjust business rating:
    ```
    python src/review_adjustment.py
    ```
6. Find the best dish names:
    ```
    python src/best_dishes.py
    ```


### Results
1. Adjusted business review rating is generated in file:
    ```
    output/adjusted_review_result.csv
    ```
   - Column ```exact_review_from_user``` lists the exact review ratings originally from Yelp. Note Yelp provides the rounded rating directly, to compare the exact original review rating, here we calculate it through averaging each individual rating from users.
   - Column ```adjusted_business_review_rating``` lists the review ratings after adjustment.
   - Columns ```[adj_based_on_food_review_count, adj_based_on_total_review_count, adj_based_on_elite_status, adj_based_on_useful_count, adj_based_on_avg_rating, adj_based_on_topic]``` are the weight adjustment for each review.
   
   Comparing the rating before and after adjustment:
   * Max rating addition: 0.28
   * Max rating reduction: 0.27
   * Average rating change: +0.017
   
   Below are the restaurants and their business id with maximum addition and reduction:
   
   | Restaurant | Rating Adjustment |
   | :---: | :---: |
   | Qwik's Chinese Bistro_46LhKfz6MPaLYoS0jegsdw | -0.271 |
   | Yunnan Garden_dVhGY-mNwTWQzK01Zxuclw | 0.284 |

    From the eventual adjusted_business_review_rating, we can get the best restaurants, here we list the top 10:

   | Restaurant | Rating Before Adjustment | Rating After Adjustment |
   | :---: | :---: | :---: |
   | China Passion_BKg8YIGX_5YyUczmBAyyCQ | 4.605 | 4.59 |
   | Phoenix Express_t_SvwKRaMyNHj2NOMvMv1Q | 4.536 | 4.571 |  
   | Veggie House_AtD6B83S4Mbmq0t7iDnUVA | 4.548 | 4.553 |
   | Kung Foo Noodle_O7_rXHN_-cFp0TgiMaNulw | 4.547 | 4.541 | 
   | Pink Pepper Asian Fusion_B1tsu2zSVlY_g1QS-z4ILg | 4.604 | 4.54 |
   | Nuro bistro_GI1F8a__wktcfj6YVjTqAA | 4.582 | 4.539 | 
   | Double 10 Mini Hot Pot_QlAPX3c0Vm2dU_w3puIAIw | 4.514 | 4.536 |
   | Simi's Cafe_W9y-Bl9030-rmfxSTmUD_g | 4.607 | 4.512 | 
   | Singing Pandas Asian Restaurant & Bar_bWucOPNoIjd8ECdiDyVq9Q | 4.558 | 4.509 |
   | Pepper Cafe_Av5toUG7wweXfiX6FW4F0A | 4.626 | 4.507 | 

2. The best dishes from the best restaurants (>4 star rating, around 100 restaurants) are listed in below file:
    ```
    output/4star_pos_reviews_most_common_trigrams
    ```
    The list is generated based on parsing the reviews with positive sentiments on the best restaurants. Below are the top dishes and the number of times they were mentioned in the review:
    ![alt text](https://github.com/yix7/mcsds_cs410_project/blob/master/output/BestDishesList.PNG)
    We also tried bi-grams for the dish names, but with bi-gram, the names are more generic, which don't represent the exact dish names, thus we used tri-gram to represent.
    
