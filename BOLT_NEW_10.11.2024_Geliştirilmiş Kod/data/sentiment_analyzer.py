from transformers import pipeline
import tweepy
import praw
import numpy as np

class SentimentAnalyzer:
    def __init__(self, twitter_api_key=None, reddit_api_key=None):
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.setup_social_media_apis(twitter_api_key, reddit_api_key)
        
    def setup_social_media_apis(self, twitter_api_key, reddit_api_key):
        if twitter_api_key:
            auth = tweepy.OAuthHandler(twitter_api_key['consumer_key'], 
                                     twitter_api_key['consumer_secret'])
            auth.set_access_token(twitter_api_key['access_token'], 
                                twitter_api_key['access_token_secret'])
            self.twitter_api = tweepy.API(auth)
            
        if reddit_api_key:
            self.reddit_api = praw.Reddit(
                client_id=reddit_api_key['client_id'],
                client_secret=reddit_api_key['client_secret'],
                user_agent=reddit_api_key['user_agent']
            )
    
    def analyze_social_sentiment(self, symbol, timeframe='1h'):
        tweets = self.get_tweets(symbol)
        reddit_posts = self.get_reddit_posts(symbol)
        
        all_texts = tweets + reddit_posts
        sentiments = self.sentiment_analyzer(all_texts)
        
        # Aggregate sentiment scores
        sentiment_scores = [s['score'] if s['label'] == 'POSITIVE' else -s['score'] 
                          for s in sentiments]
        
        return np.mean(sentiment_scores)