import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tensorflow.keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
import joblib
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
import io
import base64
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')

class TweetSentimentApp:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.sent_analyzer = SentimentIntensityAnalyzer()
        self.positive_words = self.read_word_list('positive_words.txt')
        self.negative_words = self.read_word_list('negative_words.txt')
        self.neutral_words = self.read_word_list('neutral_words.txt')
        self.constructive_words = self.read_word_list('constructive_words.txt')
        self.destructive_words = self.read_word_list('destructive_words.txt')
        self.agitative_words = set(open('agitative_words.txt').read().splitlines())
        self.max_sequence_length = 100
        # Load your logistic regression model and other necessary components here
        self.multinomial_naive_bayes_model = joblib.load('multinomial_naive_bayes_model.pkl')  # Load your logistic regression model
        self.tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Load TF-IDF vectorizer or other feature extraction methods

    def clean_text(self, text):
        text = str(text)
        text = re.sub(r'@[A-Za-z0-9]+', '', text)  # remove mentions
        text = re.sub(r'#', '', text)  # remove hashtags
        text = re.sub(r'RT[\s]+', '', text)  # remove retweets
        text = re.sub(r'https?:\/\/\S+', '', text)  # remove links
        text = re.sub(r'[^A-Za-z0-9\s]+', '', text)  # remove special characters
       
        return " ".join(nltk.word_tokenize(text.lower().strip()))
        

    def analyze_sentiment(self, text):
        
        sid = SentimentIntensityAnalyzer()
        sentiment_score = sid.polarity_scores(text)['compound']
        sentiment_tag = 'positive' if sentiment_score > 0 else ('negative' if sentiment_score < 0 else 'neutral')
        
        return sentiment_score, sentiment_tag
    
    def predict_sentiment(self, text):
        
        if isinstance(text, str):
            text = str(text)
            # Vectorize the text using the same method used during training
            text_vector = self.tfidf_vectorizer.transform([text])  # You can adjust this for your specific preprocessing
            # Predict sentiment using the logistic regression model
            sentiment_label = self.multinomial_naive_bayes_model.predict(text_vector)
            return sentiment_label[0]  # Return the predicted sentiment label
      
        return "N/A", None
    def sentiment_label(self, sentiment_score):
        try:
            sentiment_score = float(sentiment_score)
            if sentiment_score > 0:
                return 'positive'
            elif sentiment_score < 0:
                return 'negative'
            else:
                return 'neutral'
        except ValueError:
            return 'Invalid'
    def get_sentiment_score(self, tweet):
        
        if isinstance(tweet, str):
            return self.sent_analyzer.polarity_scores(tweet)['compound']
        return 0.0  # Default score for non-text values
    def calculate_positive_sentiment_percentage(self, dataframe):
        # Apply sentiment analysis to each tweet
        dataframe['sentiment_score'] = dataframe['tweets'].apply(lambda tweet: self.get_sentiment_score(tweet))
        # Categorize sentiment into tags
        dataframe['sentiment_tag'] = dataframe['sentiment_score'].apply(self.sentiment_label)
        # Group by user name and sentiment, and count the number of tweets for each combination
        sentiment_counts = dataframe.groupby(['username', 'sentiment_tag']).size().reset_index(name='count')
        # Pivot the sentiment counts dataframe to have sentiments as columns
        sentiment_pivot = sentiment_counts.pivot(index='username', columns='sentiment_tag', values='count').fillna(0)
        # Calculate the percentage of positive sentiment for each user
        sentiment_pivot['total_tweets'] = sentiment_pivot.sum(axis=1)
        sentiment_pivot['positive_percentage'] = (sentiment_pivot['positive'] / sentiment_pivot['total_tweets']) * 100
        return sentiment_pivot[['positive_percentage']]
    
    def plot_sentiments_by_politician(self, df):
        
        politician_sentiments = df.groupby('username')['sentiment_tag'].value_counts(normalize=True).unstack().fillna(0)
        # Plot bar chart for sentiments
        politician_sentiments.plot(kind='bar', figsize=(10, 6))
        plt.xlabel('Politician')
        plt.ylabel('Percentage of Sentiments')
        plt.title('Sentiments by Politician')
        plt.xticks(rotation=45)
        st.pyplot()
        
    def find_most_active_politician(self, df):

        politician_tweet_counts = df['username'].value_counts()
        most_active_politician = politician_tweet_counts.idxmax()
        num_tweets = politician_tweet_counts.max()
        return most_active_politician, num_tweets
    # =============================================================================================
    def read_word_list(self, filename):
        with open(filename, 'r') as file:
            return [line.lower().strip() for line in file]
    # =============================================================================================
    def analyze_tweets_sentiment(self, dataframe):
        mentions = []
        for index, row in dataframe.iterrows():
            username = row['username']
            tweet = str(row['tweets'])  # Convert to string to handle potential non-string values

            blob = TextBlob(tweet)
            for word in blob.words:
                if word in self.positive_words:
                    mentions.append((username, word, 'positive'))
                elif word in self.negative_words:
                    mentions.append((username, word, 'negative'))
                elif word in self.neutral_words:
                    mentions.append((username, word, 'neutral'))
                # elif word in self.agitative_words:
                #     mentions.append((username, word, 'agitative'))
        
        return mentions
    
    def sentiment_count(self, dataframe):
        
        # Group the dataframe by user name and count the number of tweets
        tweet_count_df = dataframe.groupby('sentiment_tag')['tweets'].count().reset_index()
        
        # Find the user with the highest tweet count
        user_with_highest_tweet_count = tweet_count_df.loc[tweet_count_df['tweets'].idxmax()]
        
        # Print the user name and highest tweet count
        user_name = user_with_highest_tweet_count['sentiment_tag']
        tweet_count = user_with_highest_tweet_count['tweets']
        print(f"Overall Sentiment Count: {user_name}, Tweet Count: {tweet_count}")
        
        # Plot tweet count of each leader
        plt.bar(tweet_count_df['sentiment_tag'], tweet_count_df['tweets'], color=['red', 'green', 'blue'], width=0.5)
        plt.xlabel('Sentiment')
        plt.ylabel('Sentiment Count')
        plt.title('Overall Sentiment Count')
        plt.xticks(rotation=45)
        # Show the tweet count on top of each bar
        for i, count in enumerate(tweet_count_df['tweets']):
            plt.text(i, count + 5, str(count), ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        st.pyplot()
    # =============================================================================================

    def plot_word_frequency_by_user(self, df):
        user_word_frequency = {}

        for index, row in df.iterrows():
            username = row['username']
            tweet = str(row['tweets'])  # Convert to string to handle potential non-string values

            blob = TextBlob(tweet)
            for word in blob.words:
                if word in self.positive_words or word in self.negative_words or word in self.neutral_words:
                    sentiment = ''
                    if word in self.positive_words:
                        sentiment = 'positive'
                    elif word in self.negative_words:
                        sentiment = 'negative'
                    else:
                        sentiment = 'neutral'
                        
                    if username not in user_word_frequency:
                        user_word_frequency[username] = {
                            'positive': {},
                            'negative': {},
                            'neutral': {}
                        }
                    if word not in user_word_frequency[username][sentiment]:
                        user_word_frequency[username][sentiment][word] = 0
                    user_word_frequency[username][sentiment][word] += 1

        for user, sentiment_freq in user_word_frequency.items():
            for sentiment, word_freq in sentiment_freq.items():
                self.plot_word_frequency(user, word_freq, sentiment)
    
    def plot_word_frequency(self, user, word_freq, sentiment):
        word_freq_df = pd.DataFrame(word_freq.items(), columns=['Word', 'Frequency'])
        plt.figure(figsize=(10, 6))
        plt.bar(word_freq_df['Word'], word_freq_df['Frequency'], color='green')
        plt.xlabel('Word')
        plt.ylabel('Frequency')
        plt.title(f'{sentiment.title()} Word Frequency for User: {user}')
        plt.xticks(rotation=45)
        st.pyplot()
    # =============================================================================================
    def calculate_sentiment_percentages(self, df):
        sentiment_percentages = {}

        for index, row in df.iterrows():
            username = row['username']
            tweet = str(row['tweets'])  # Convert to string to handle potential non-string values

            blob = TextBlob(tweet)
            positive_count = 0
            negative_count = 0
            neutral_count = 0

            for word in blob.words:
                if word in self.positive_words:
                    positive_count += 1
                elif word in self.negative_words:
                    negative_count += 1
                elif word in self.neutral_words:
                    neutral_count += 1

            if username not in sentiment_percentages:
                sentiment_percentages[username] = {
                    'positive': 0,
                    'negative': 0,
                    'neutral': 0
                }

            total_words = positive_count + negative_count + neutral_count
            if total_words > 0:
                sentiment_percentages[username]['positive'] += (positive_count / total_words)
                sentiment_percentages[username]['negative'] += (negative_count / total_words)
                sentiment_percentages[username]['neutral'] += (neutral_count / total_words)

        return sentiment_percentages
# =============================================================================================
    def find_most_patriotic_politician(self, df):
        politician_sentiment_count = {}

        for index, row in df.iterrows():
            username = row['username']
            sentiment = row['sentiment_tag']

            if sentiment == 'positive':
                if username not in politician_sentiment_count:
                    politician_sentiment_count[username] = 0
                politician_sentiment_count[username] += 1

        if politician_sentiment_count:
            most_patriotic_politician = max(politician_sentiment_count, key=politician_sentiment_count.get)
            return most_patriotic_politician, politician_sentiment_count[most_patriotic_politician]
        else:
            return None, None
# =============================================================================================
    def find_user_with_highest_tweet_count(self, dataframe):
        # Group the dataframe by user name and count the number of tweets
        tweet_count_df = dataframe.groupby('username')['tweets'].count().reset_index()
        
        # Find the user with the highest tweet count
        user_with_highest_tweet_count = tweet_count_df.loc[tweet_count_df['tweets'].idxmax()]
        
        # Print the user name and highest tweet count
        user_name = user_with_highest_tweet_count['username']
        tweet_count = user_with_highest_tweet_count['tweets']
        st.write(f"#### Highest Tweet Count: {user_name}, Tweet Count: {tweet_count}")
        
        # Plot tweet count of each leader
        plt.bar(tweet_count_df['username'], tweet_count_df['tweets'], color=['red', 'green', 'blue'], width=0.3)
        plt.xlabel('Politician')
        plt.ylabel('Tweet Count')
        plt.title('Tweet Count of Each Leader')
        plt.xticks(rotation=45)
        # Show the tweet count on top of each bar
        for i, count in enumerate(tweet_count_df['tweets']):
            plt.text(i, count + 5, str(count), ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        st.pyplot()
# =============================================================================================
    def detect_agitation(self, tweet):
        # Search for agitation keywords in the tweet text
        for word in self.agitative_words:
            if re.search(r'\b' + word + r'\b', tweet, re.IGNORECASE):
                return True
        return False
    #     with open(filename, 'r') as file:
    #         return [line.lower().strip() for line in file]
        

    def analyze_tweets_for_agitation(self, dataframe):
        agitating_politicians = {}

        for index, row in dataframe.iterrows():
            username = row['username']
            tweet = row['tweets']

            # Handle non-string values in tweet column
            if not isinstance(tweet, str):
                continue  # Skip this row

            # Apply sentiment analysis
            blob = TextBlob(tweet)
            sentiment_score = blob.sentiment.polarity

            # Check for agitation keywords
            agitation_keywords = self.get_agitation_keywords(tweet)
            has_agitation = any(keyword in tweet.lower() for keyword in agitation_keywords)

            # Determine if politician is inciting agitation
            if sentiment_score < 0 and has_agitation:
                if username in agitating_politicians:
                    agitating_politicians[username]['count'] += len(agitation_keywords)
                    agitating_politicians[username]['keywords'].update(agitation_keywords)
                else:
                    agitating_politicians[username] = {
                        'count': len(agitation_keywords),
                        'keywords': agitation_keywords
                    }

        return agitating_politicians
    def find_most_agitative_politician(self, agitating_politicians):
        most_agitative_politician = max(agitating_politicians, key=lambda x: agitating_politicians[x]['count'])
        return most_agitative_politician
    
    def get_agitation_keywords(self, tweet):
        agitation_keywords = set()

        for word in self.agitative_words:
            if re.search(r'\b' + word + r'\b', tweet, re.IGNORECASE):
                agitation_keywords.add(word)

        return agitation_keywords
    def predict_single_user_sentiment(self, user_input):
        # Predict sentiment for a single user input
        cleaned_input = self.clean_text(user_input)
        sentiment_score, sentiment_tag = self.analyze_sentiment(cleaned_input)
        predicted_sentiment_label = self.sentiment_label(sentiment_score)

        return cleaned_input, sentiment_score, sentiment_tag, predicted_sentiment_label
    # ====================================================================================
    # Phase3
    def most_constructive_and_destructive(self, dataframe, constructive_words, destructive_words):
        politician_scores = {}

        for index, row in dataframe.iterrows():
            username = row['username']
            tweet = str(row['tweets'])

            constructive_count = sum(1 for word in tweet.split() if word in constructive_words)
            destructive_count = sum(1 for word in tweet.split() if word in destructive_words)

            if username not in politician_scores:
                politician_scores[username] = {
                    'constructive_score': 0,
                    'destructive_score': 0
                }

            politician_scores[username]['constructive_score'] += constructive_count
            politician_scores[username]['destructive_score'] += destructive_count

        most_constructive_politician = max(politician_scores, key=lambda x: politician_scores[x]['constructive_score'])
        most_destructive_politician = max(politician_scores, key=lambda x: politician_scores[x]['destructive_score'])

        return most_constructive_politician, most_destructive_politician
    # ====================================================================================
    # WordCloud Generation
    # ====================================================================================
    def generate_wordclouds_for_each_user(self, dataframe):
        for sentiment in ['positive', 'negative', 'neutral']:
            for username in dataframe['username'].unique():
                words = self.get_words_by_sentiment(dataframe, username, sentiment)
                if words:
                    self.plot_wordcloud(username, sentiment, words)

    def get_words_by_sentiment(self, dataframe, username, sentiment):
        words = ' '.join(dataframe[(dataframe['username'] == username) & (dataframe['sentiment_tag'] == sentiment)]['cleaned_text'])
        return words

    def plot_wordcloud(self, username, sentiment, words):
        wordcloud = WordCloud(width=800, height=400, background_color='black').generate(words)

        # Convert the word cloud image to a base64-encoded string
        image_stream = io.BytesIO()
        wordcloud.to_image().save(image_stream, format='PNG')
        image_str = base64.b64encode(image_stream.getvalue()).decode()

        # Display the word cloud image using st.image()
        st.image(f"data:image/png;base64,{image_str}", caption=f'{sentiment.title()} Word Cloud for {username}', use_column_width=True)
    # ===================================================================================================================
    def run(self):
        st.title('Tweet Sentiment Analysis')
        st.write("### Analyze Sentiment for a Single User Input")
        user_input = st.text_area("Enter a tweet for sentiment analysis:")
        if st.button("Analyze Sentiment"):
            if user_input:
                cleaned_input, sentiment_score, sentiment_tag, predicted_sentiment_label = self.predict_single_user_sentiment(user_input)
                single_user_sentiment_df = pd.DataFrame({'User Input': [cleaned_input],
                                                         'Sentiment Score': [sentiment_score],
                                                         'Sentiment Tag': [sentiment_tag],
                                                         'Predicted Sentiment Label': [predicted_sentiment_label],
                                                         # "agitated words": [self.get_agitation_keywords(user_input)],
        })
                st.dataframe(single_user_sentiment_df)
            else:
                st.warning("Please enter a tweet for sentiment analysis.")    

        st.write("### Upload a file to Analyze Sentiment")
        st.write("Column name should be 'tweets' and 'username' for multiuser tweets sentiment")
        uploaded_file = st.file_uploader('Upload a file', type=['csv', 'xlsx'])

        if uploaded_file is not None:
            if uploaded_file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':  # XLSX
                df = pd.read_excel(uploaded_file)
            else:  # CSV
                df = pd.read_csv(uploaded_file)

            
            # =========================================================================================
            df['cleaned_text'] = df['tweets'].apply(self.clean_text)
            df['sentiment_score'], df['sentiment_tag'] = zip(*df['cleaned_text'].apply(self.analyze_sentiment))
            df['predicted_sentiment'] = df['cleaned_text'].apply(self.predict_sentiment)
            df['predicted_sentiment_label'] = df['predicted_sentiment'].apply(self.sentiment_label)
            # positive_percentage = (df['sentiment_tag'] == 'positive').mean() * 100
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.figure(figsize=(8, 6))
            plt.hist(df['sentiment_score'], bins=20, color='green', alpha=0.7)
            plt.xlabel('Sentiment Score')
            plt.ylabel('Frequency')
            plt.title('Sentiment Distribution')
            st.pyplot()
            st.write("### Data Content")
            st.dataframe(df)

            df_with_topics = analyzer.perform_topic_modeling_and_analysis(df)

            most_active_politician, num_tweets = self.find_most_active_politician(df)

            mentions_df = pd.DataFrame(self.analyze_tweets_sentiment(df),columns=['User', 'Mentioned Word', 'Sentiment'])

            # Display the combined DataFrame using st.dataframe()
            st.write("### Positive, Negative, Neutral Mentions by each Politician")
            st.dataframe(mentions_df)
            # Plot word frequency for each user
            self.plot_word_frequency_by_user(df)
            st.write('### Sentiment Analysis Results')
            st.write(df)
            # Display positive, negative, neutral sentiment graph
            # self.plot_sentiments_by_politician(df)
            self.calculate_positive_sentiment_percentage(df)
            
            # Calculate sentiment percentages
            sentiment_percentages = self.calculate_sentiment_percentages(df)
            # Convert sentiment percentages to dataframe
            sentiment_percentages_df = pd.DataFrame(sentiment_percentages).transpose()
            # Plot sentiment percentages
            sentiment_percentages_df.plot(kind='bar', figsize=(10, 6))
            plt.xlabel('Politician')
            plt.ylabel('Percentage of Sentiments')
            plt.title('Sentiments')
            plt.xticks(rotation=45)
            st.pyplot()

            # Find the highest tweet count and plot it
            self.find_user_with_highest_tweet_count(df)
            # Display sentiment percentages dataframe
            st.write('### Sentiment Percentages')
            st.dataframe(sentiment_percentages_df)

            # Find most active politician
            st.write(f"#### Most Active Politician: {most_active_politician} with {num_tweets} tweets")
            # Find most patriotic politician
            most_patriotic_politician, patriotic_count = self.find_most_patriotic_politician(df)

            if most_patriotic_politician:
                st.write(f"#### Most Patriotic Politician: {most_patriotic_politician}")
                # st.write(f"### Positive Sentiment Mentions: {patriotic_count}")
            else:
                st.write("No positive sentiment mentions found.")

            # Detect politicians inciting agitation
            # agitating_politicians = self.analyze_tweets_for_agitation(df)
            agitating_politicians = self.analyze_tweets_for_agitation(df)

            # Display politicians inciting agitation in a table
            st.write("### Politicians Inciting Agitation:")
            agitating_table_data = []
            for politician, data in agitating_politicians.items():
                keywords = ", ".join(data['keywords'])
                frequency = data['count']
                agitating_table_data.append((politician, keywords, frequency))

            st.table(pd.DataFrame(agitating_table_data, columns=['Politician', 'Agitation Keywords', 'Keyword Frequency']))
            # Call the function to find the politicians
            # Usage example
            most_constructive, most_destructive = self.most_constructive_and_destructive(df, self.constructive_words, self.destructive_words)
            st.write("### Most Constructive Politician:", most_constructive)
            st.write("### Most Destructive Politician:", most_destructive)
            
            st.write("## Politician words for each others.")
            # Print the dataframe with topics assigned
            st.dataframe(df_with_topics[['username', 'top_topic_words']])
            st.write("## Behavior Analysis via Sentiments:")
            self.plot_sentiments_by_politician(df)

            # Plot word clouds for each user
            st.write("## Word Clouds for Each User")
            self.generate_wordclouds_for_each_user(df)
# =====================================================================================================
# ================================================================================================================================
class TopicModelingAnalyzer:
    def __init__(self, tfidf_vectorizer_path, num_topics=3):
        self.tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)  # Load the TF-IDF vectorizer
        self.num_topics = num_topics
        self.lda = LatentDirichletAllocation(n_components=self.num_topics, random_state=42)

    def perform_topic_modeling_and_analysis(self, df):
        # Fill missing values in the 'tweets' column with an empty string
        df['tweets'].fillna('', inplace=True)
        
        # Transform the text data using the loaded vectorizer
        tfidf_matrix = self.tfidf_vectorizer.transform(df['tweets'])

        # Fit LDA for topic modeling
        self.lda.fit(tfidf_matrix)
        
        # Get the topics for each tweet
        df['topics'] = self.lda.transform(tfidf_matrix).argmax(axis=1)
        
        # Get the top words for each topic
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        top_words_per_topic = []
        for topic_idx, topic in enumerate(self.lda.components_):
            top_words_idx = topic.argsort()[:-10 - 1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            top_words_per_topic.append(top_words)

        # Add a column with the top topic words for each tweet
        df['top_topic_words'] = [top_words_per_topic[topic] for topic in df['topics']]
        
        return df

class MultiuserTweetSentimentApp(TweetSentimentApp):
    def analyze_multiuser_behavior(self, dataframe):
        # Analyze sentiment scores
        sentiment_scores = dataframe['cleaned_text'].apply(self.analyze_sentiment)

        # Extract sentiment score and sentiment tag from the tuple (if it is a tuple)
        sentiment_scores = sentiment_scores.apply(lambda x: x if isinstance(x, tuple) else (x, 'Unknown'))

        # Unpack the tuple into sentiment score and sentiment tag
        dataframe['sentiment_score'], dataframe['sentiment_tag'] = zip(*sentiment_scores)

        # Apply sentiment_label function to the 'sentiment_score' column
        dataframe['predicted_sentiment_label'] = dataframe['sentiment_score'].apply(self.sentiment_label)

        # Calculate sentiment distribution and add it to the DataFrame
        sentiment_distribution = dataframe.groupby(['username', 'sentiment_tag']).size()  # Group by user and sentiment tag
        sentiment_distribution = sentiment_distribution.reset_index(name='count')
        total_tweets = sentiment_distribution.groupby('username')['count'].sum()
        sentiment_distribution['percentage'] = (sentiment_distribution['count'] / total_tweets) * 100

        # Calculate topic distribution and add it to the DataFrame (if 'topics' column exists)
        if 'topics' in dataframe.columns:
            topic_distribution = dataframe.groupby(['username', 'topics']).size()  # Group by user and topic
            topic_distribution = topic_distribution.reset_index(name='count')
            total_topics = topic_distribution.groupby('username')['count'].sum()
            topic_distribution['percentage'] = (topic_distribution['count'] / total_topics) * 100

            return dataframe, sentiment_distribution, topic_distribution

        return dataframe, sentiment_distribution


    def plot_multiuser_behavior(self, behavioral_analysis):
        # Plot Sentiment Distribution for Each User
        for user, sentiment_dist in behavioral_analysis['sentiment_distribution'].groupby(level=0):
            sentiment_dist.plot(kind='bar', figsize=(6, 4))
            plt.xlabel('Sentiment')
            plt.ylabel('Percentage')
            plt.title(f'Sentiment Distribution for {user}')
            st.pyplot()  # Use st.pyplot() to display the plot in Streamlit

        # Plot Topic Distribution for Each User
        for user, topic_dist in behavioral_analysis['topic_distribution'].groupby(level=0):
            topic_dist.plot(kind='bar', figsize=(6, 4))
            plt.xlabel('Dominant Topic')
            plt.ylabel('Percentage')
            plt.title(f'Topic Distribution for {user}')
            st.pyplot()  # Use st.pyplot() to display the plot in Streamlit

    def run_multiuser_analysis(self, dataframe):
        if 'username' not in dataframe.columns or 'sentiment_tag' not in dataframe.columns:
            st.warning("The input dataframe does not contain necessary columns.")
            return

        dataframe, sentiment_distribution, topic_distribution = self.analyze_multiuser_behavior(dataframe)

        # Plot Sentiment Distribution for Each User
        for user, sentiment_dist in sentiment_distribution.groupby('username'):
            fig, ax = plt.subplots()
            ax.bar(sentiment_dist['sentiment_tag'], sentiment_dist['percentage'])
            ax.set_xlabel('Sentiment')
            ax.set_ylabel('Percentage')
            ax.set_title(f'Sentiment Distribution for {user}')
            st.pyplot(fig)

        # Plot Topic Distribution for Each User (if 'topics' column exists in the DataFrame)
        if 'topics' in dataframe.columns:
            for user, topic_dist in topic_distribution.groupby('username'):
                fig, ax = plt.subplots()
                ax.bar(topic_dist['topics'], topic_dist['percentage'])
                ax.set_xlabel('Dominant Topic')
                ax.set_ylabel('Percentage')
                ax.set_title(f'Topic Distribution for {user}')
                st.pyplot(fig)


    
# =============================================================================================================================================

if __name__ == '__main__':
    app = MultiuserTweetSentimentApp()
    analyzer = TopicModelingAnalyzer(tfidf_vectorizer_path='tfidf_vectorizer.pkl')
    # app.run_multiuser_analysis(SentimentIntensityAnalyzer.run.df)
    app.run()
