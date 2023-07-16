#
#
# main() will be run when you invoke this action
#
# @param Cloud Functions actions accept a single parameter, which must be a JSON object.
#
# @return The output of this action, which must be a JSON object.
#
#
import sys
import json
import requests
import numpy as np
import pandas as pd
import ibm_boto3
from ibm_botocore.client import Config, ClientError
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_json(obj):
    return {k: v for k, v in obj.items() if v is not None and k != '__ow_method' and k != '__ow_headers' and k != '__ow_path'}


def main(params):

    # Access dataset in CloudObjectStorage
    cloud_object_storage_cos_standard_4bg_credentials = {
        "apikey": "P0HpH66QCyfJohxY3IOrjJZu2z-l6FQbRH5dhUHGQq3e",
        "endpoints": "https://control.cloud-object-storage.cloud.ibm.com/v2/endpoints",
        "iam_apikey_description": "Auto-generated for key crn:v1:bluemix:public:cloud-object-storage:global:a/15743b958ef743da8df43455255697ae:b14b5f12-79ad-43f9-871c-355321f19791:resource-key:ed07f233-5aef-4ced-aa30-b24617df6363",
        "iam_apikey_name": "cloud-object-storage-cos-standard-4bg",
        "iam_role_crn": "crn:v1:bluemix:public:iam::::serviceRole:Writer",
        "iam_serviceid_crn": "crn:v1:bluemix:public:iam-identity::a/15743b958ef743da8df43455255697ae::serviceid:ServiceId-e329115c-a0c2-4649-8381-6bd9ce097f70",
        "resource_instance_id": "crn:v1:bluemix:public:cloud-object-storage:global:a/15743b958ef743da8df43455255697ae:b14b5f12-79ad-43f9-871c-355321f19791::"
    }
    
    cos = ibm_boto3.client("s3",
        ibm_api_key_id = cloud_object_storage_cos_standard_4bg_credentials['apikey'],
        ibm_service_instance_id = cloud_object_storage_cos_standard_4bg_credentials['iam_serviceid_crn'],
        config = Config(signature_version = "oauth"),
        endpoint_url = 'https://s3.us-east.cloud-object-storage.appdomain.cloud'
    )

    # Transform data into pandas dataframe
    response = cos.get_object(Bucket='recipe-data-1', Key='food_df_cleaned.csv')
    filtered_food_df = pd.read_csv(response['Body'])
    
    filtered_food_df['description'] = filtered_food_df['description'].fillna('')
    filtered_food_df['name'] = filtered_food_df['name'].fillna('')

    # intialize final output string
    returnString = ""
    
    # # takes in parameters from Watson Assistant and loads them into a JSON
    data = json.dumps(clean_json(params))
    data1 = json.loads(data)
    
    # filters dataset based on JSON parameters
    if 'fruit' in data1:
        fruit = data1['fruit']
        filtered_food_df = filtered_food_df[filtered_food_df['tags'].str.contains(fruit) == False]
        if fruit == "citrus":
            filtered_food_df = filtered_food_df[filtered_food_df['ingredients'].str.contains('orange|lime|lemon|grapefruit') == False]

    if 'vegan' in data1:
        vegan = data1['vegan']
        filtered_food_df = filtered_food_df[filtered_food_df['tags'].str.contains(vegan)]
    
    if 'vegetarian' in data1:
        vegetarian = data1['vegetarian']
        filtered_food_df = filtered_food_df[filtered_food_df['tags'].str.contains(vegetarian)]
    
    if 'kosher' in data1:
        kosher = data1['kosher']
        filtered_food_df = filtered_food_df[filtered_food_df['tags'].str.contains(kosher)]
    
    if 'nuts' in data1:
        nut_free = data1['nuts']
        filtered_food_df = filtered_food_df[filtered_food_df['tags'].str.contains(nut_free) == False]
    
    if 'lactose_intolerant' in data1:
        lactose_intolerant = data1['lactose_intolerant']
        filtered_food_df = filtered_food_df[filtered_food_df['tags'].str.contains("lactose" or "dairy-free")]
    
    if 'fish' in data1:
        fish = data1['fish']
        filtered_food_df = filtered_food_df[filtered_food_df['tags'].str.contains(fish) == False]
        if fish == "shellfish":
            filtered_food_df = filtered_food_df[filtered_food_df['ingredients'].str.contains('shrimp') == False]
    
    if 'gluten_free' in data1:
        gluten_free = data1['gluten_free']
        filtered_food_df = filtered_food_df[filtered_food_df['tags'].str.contains(gluten_free)]
    
    if 'nutrition_restriction' in data1:
        nutrition_restriction = data1['nutrition_restriction']
        if nutrition_restriction == 'low_cal':
            filtered_food_df = filtered_food_df[filtered_food_df['calories (#)'] < 174.40]
        elif nutrition_restriction == 'low_fat':
            filtered_food_df = filtered_food_df[filtered_food_df['total fat (g)'] < 6.00]
        elif nutrition_restriction == 'low_sugar':
            filtered_food_df = filtered_food_df[filtered_food_df['sugar (g)'] < 4.50]
        elif nutrition_restriction == 'low_sodium':
            filtered_food_df = filtered_food_df[filtered_food_df['sodium (mg)'] < 115.00]
        elif nutrition_restriction == 'high_protein':
            filtered_food_df = filtered_food_df[filtered_food_df['protein (g)'] > 25.50]
        elif nutrition_restriction == 'low_carb':
            filtered_food_df = filtered_food_df[filtered_food_df['carbohydrates (g)'] < 11.00]
    
    if 'meal_bev_type' in data1:
        meal_bev_type = data1['meal_bev_type']
        filtered_food_df = filtered_food_df[filtered_food_df['tags'].str.contains(meal_bev_type)]
    
    if 'holiday_season_calendarEvent_desc' in data1:
        holiday_season_calendarEvent_desc = data1['holiday_season_calendarEvent_desc']
        filtered_food_df = filtered_food_df[filtered_food_df['tags'].str.contains(holiday_season_calendarEvent_desc)]
    
    if 'cuisine' in data1:
        cuisine = data1['cuisine']
        filtered_food_df = filtered_food_df[filtered_food_df['tags'].str.contains(cuisine)]
    
    if 'time_limit' in data1:
        time_limit = data1['time_limit']
        if time_limit == '15-minutes-or-less':
            filtered_food_df = filtered_food_df[filtered_food_df['minutes'] < 15]
        elif time_limit == '30-minutes-or-less':
            filtered_food_df = filtered_food_df[filtered_food_df['minutes'] < 30]
        elif time_limit == '60-minutes-or-less':
            filtered_food_df = filtered_food_df[filtered_food_df['minutes'] < 60]
        elif time_limit == '4-hours-or-less':
            filtered_food_df = filtered_food_df[filtered_food_df['minutes'] < 240]
    
    if 'liquors' in data1:
        liquors = data1['liquors']
        filtered_food_df = filtered_food_df[filtered_food_df['ingredients'].str.contains(liquors)]
    
    if 'cocktail_type' in data1:
        cocktail_type = data1['cocktail_type']
        filtered_food_df = filtered_food_df[filtered_food_df['name'].str.contains(cocktail_type)]
        
    
    # Define the input from the user
    if 'misc_description' in data1:
        misc_description = data1['misc_description']
    elif 'liquors' in data1:
        misc_description = liquors
    elif 'cocktail_type' in data1:
        misc_description = cocktail_type

    # Preprocess the input by removing punctuation and converting to lowercase
    input_ingredients = misc_description.lower().replace(",", "")
    
    # Use CountVectorizer to create a matrix of word counts for each recipe name
    vectorizer = CountVectorizer(stop_words="english")
    X = vectorizer.fit_transform(filtered_food_df["name"])
    
    # Convert the input into a matrix of ingredient counts
    input_matrix = vectorizer.transform([input_ingredients])
    
    # Calculate the cosine similarity between the input and each recipe
    similarity_scores = cosine_similarity(input_matrix, X)[0]
    
    # Get the index of the recipe with the highest similarity score
    most_similar_indices = np.argsort(similarity_scores)[::-1][:3]
    
    # Get the title of the most similar 
    most_similar_title = list(filtered_food_df.iloc[most_similar_indices]["name"])
    most_similar_ingredients = list(filtered_food_df.iloc[most_similar_indices]["ingredients"])
    most_similar_steps = list(filtered_food_df.iloc[most_similar_indices]["steps"])
    
    return {'sim_score_1': 'Recipe 1: Confidence Level -' + ' ' + str(similarity_scores[most_similar_indices[0]]*100) + '%',
            'recipe_1': 'Recipe Name -' + ' ' + most_similar_title[0],
            'ingr_1': 'Ingredients:' + ' ' + most_similar_ingredients[0],
            'steps_1': 'Steps:' + ' ' + most_similar_steps[0],
            'sim_score_2': 'Recipe 2: Confidence Level -' + ' ' + str(similarity_scores[most_similar_indices[1]]*100) + '%',
            'recipe_2': 'Recipe Name -' + ' ' + most_similar_title[1],
            'ingr_2': 'Ingredients:' + ' ' + most_similar_ingredients[1],
            'steps_2': 'Steps:' + ' ' + most_similar_steps[1],
            'sim_score_3': 'Recipe 3: Confidence Level -' + ' ' + str(similarity_scores[most_similar_indices[2]]*100) + '%',
            'recipe_3': 'Recipe Name -' + ' ' + most_similar_title[2],
            'ingr_3': 'Ingredients:' + ' ' + most_similar_ingredients[2],
            'steps_3': 'Steps:' + ' ' + most_similar_steps[2]
            }
