import requests
import praw
import os
import logging
import sys

IMAGE_FORMAT = ['jpeg', 'jpg', 'png', 'tiff']
SUBREDDIT_LIST = ['annakendrick', 'beards', 'emmawatson', 'headshots', 'aubreyplaza', 'classywomenofcolor', 
'emiliaclarke', 'firstimpression', 'guysinglasses', 'kpics', 'oldschoolcool', 'model', 'oliviawilde', 
'palebeauties', 'rateme', 'redheads', 'roastme' , 'freefolk', 'gameofthrones', 'tomhiddleston', 'katyperry', 'doppleganger',
 'walmartcelebrities', 'ScarlettJohansson', 'ChrisHemsworth', 'robertdowneyjr', 'TinyTrumps', 'sophieturner', 'DannyDeVito',
 'MakeupAddiction', 'faces', 'hittableFaces', 'trashy', 'cosplay', 'PrettyGirls', 'Pareidolia', 'DunderMifflin',
 'transpassing', 'LadyBoners', 'RandomActsofMakeup', 'portraits', 'redditgetsdrawn', 'HumanPorn', 'Colorization']

PATH_TO_DATA_LOCATION_WINDOWS = "E:\img_data"
PATH_TO_DATA_LOCATION_UNIX = "/img_data"

img_list = {}

def get_img_url(subreddit_name, limit = 500):
    reddit = praw.Reddit(client_id='ENTER_REDDIT_API_CREDENTIALS',
                     client_secret='ENTER_REDDIT_API_CREDENTIALS',
                     user_agent='image_crawler_v0.0')

    try:
        count = 0
        submissions = reddit.subreddit(subreddit_name).top(limit = limit)

        for submission in submissions:
            file_name = subreddit_name + '_' + submission.url.split('/')[-1]
            if file_name.split('.')[-1] not in IMAGE_FORMAT:
                continue 
            else:
                img_list[file_name] = submission.url
                count +=1
        
        print("{} query successfully - {} results".format(subreddit_name, count), file=open("url_retrieval_results.txt", "a"))

    except Exception:
        print("{} does not exist".format(subreddit_name), file=open("url_retrieval_results.txt", "a"))
        pass

def download_img(subreddit_list):
    for subreddit in subreddit_list:
        get_img_url(subreddit)
    
    error_count = 0
    for key, value in img_list.items():
        try:
            r = requests.get(value)
        except Exception:
            error_count += 1
            pass
        else:
            file_path = os.path.join(PATH_TO_DATA_LOCATION_WINDOWS, key)
            with open(file_path, mode = 'wb') as f:
                f.write(r.content)
    
    print("Number of dead URLs: {}".format(error_count))

if __name__ == '__main__':
    download_img(SUBREDDIT_LIST)

