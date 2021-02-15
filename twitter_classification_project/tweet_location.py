import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


#Imports random tweets and assign for classifiers
#Classifying using language: Naive Bayes Classifier
all_tweets = pd.read_json("random_tweets.json", lines=True)

print(len(all_tweets))
print(all_tweets.columns)
print(all_tweets.loc[0]['text'])
print(all_tweets.loc[0]['user']['location'])

#Imports location tweets and assign for classifiers
#Classifying using language: Naive Bayes Classifier
new_york_tweets = pd.read_json("new_york.json", lines=True)
london_tweets = pd.read_json("london.json", lines=True)
paris_tweets = pd.read_json("paris.json", lines=True)
print(len(london_tweets))
print(len(paris_tweets))

#Assigning variables
#Classifying using language: Naive Bayes Classifier
new_york_text = new_york_tweets["text"].tolist()
london_text = london_tweets["text"].tolist()
paris_text = paris_tweets["text"].tolist()

#Combining data and creating training labels
#Classifying using language: Naive Bayes Classifier
all_tweets = new_york_text + london_text + paris_text
labels = [0] * len(new_york_text) + [1] * len(london_text) + [2] * len(paris_text)

#Creating a training and test Set
train_data, test_data, train_labels, test_labels = train_test_split(all_tweets, labels, test_size = 0.2, random_state = 1)
print(len(train_data))
print(len(test_data))

#Create Count Vectors
counter = CountVectorizer()
counter.fit(train_data)
train_counts = counter.transform(train_data)
test_counts = counter.transform(test_data)

print(train_data[3])
print(train_counts[3])

#Train and Test the Naive Bayes Classifier
classifier = MultinomialNB()
classifier.fit(train_counts, train_labels)
predictions = classifier.predict(test_counts)

#Evaluation of model
print(accuracy_score(test_labels, predictions))
#Test 
tweet = "Hello World!"
tweet_counts = counter.transform([tweet])
print(classifier.predict(tweet_counts))


