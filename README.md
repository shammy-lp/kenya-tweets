# Kenya Tweet Classifier: Detecting Real-time Hate Speech and Misinformation
Data science - Fulltime Remote

Timeline: 19th March - 6th May

# Project By: Foresight Analytica
## Group Members
1.Morgan Abukuse

2.Dennis Mwania

3.Linet Patriciah

4.Precious Kalia

5.Felista Kiptoo

# Introduction 
## Project Overview
In this project, we take a look at the increasing challenge of hate speech and misinformation on Kenyan social media, particularly on platforms like Twitter. With the rapid rise in internet and smartphone access, millions of Kenyans now engage online daily. While this has opened up new channels for communication, it has also created room for the spread of harmful content, often targeting individuals or groups based on tribe, gender, political affiliation, or religion.

This growing issue poses real social threats, especially during elections, public unrest, or national discussions where misinformation and hate can escalate tensions, incite violence, or spread fear. To help tackle this, we have developed a real-time tweet classification system that uses natural language processing (NLP) and machine learning techniques to automatically detect and flag tweets containing hate speech or misinformation.

We aim to build a system that helps moderate content in real-time, supports fact-checking efforts, and promotes a safer digital space for all users in Kenya. By identifying toxic content early, this project supports peace-building, public awareness, and more responsible use of social media.

## Bussiness Problem 
In Kenya, Twitter has become a powerful space where people share opinions and talk about national issues. It gives citizens a voice and helps them organize around important topics like politics, health, and education. However, this freedom has also led to problems. There’s been a rise in hate speech and misinformation, especially during elections or times of political tension. Tweets that attack certain tribes, spread false news, or incite violence can go viral very fast.

This creates a serious challenge. Misinformation can mislead the public and hate speech can fuel division and even lead to real-world conflict. Kenyans need a way to stay safe and informed online. Right now, there’s no real-time system that detects harmful tweets in Kenya. That means hate speech and false information can spread without being stopped in time.

This project aims to build a tweet classifier that detects hate speech and misinformation in real time. It will help protect people online, reduce digital harm, and support peaceful, truthful conversations in Kenya’s online space.

## Project Objectives
1. Detects hate speech in tweets using Natural Language Processing (NLP)
2. Flags misinformation by matching tweets with verified claims from PesaCheck and AfricaCheck
3. Provides a real-time dashboard for tweet input, analysis, and classification
4. Empowers key institutions (NCIC, CAK, IEBC, DCI, media) to respond proactively
5. Supports peace, truth, and civic engagement in Kenya’s online spaces

## Key Stakeholders
1. IEBC (Independent Electoral and Boundaries Commission) – Track digital threats to election integrity

2. Fact-checking Organizations – e.g., PesaCheck and AfricaCheck

3. Journalists & Media Houses – Avoid unknowingly amplifying misinformation

4. Government and Regulatory Bodies
Agencies like DCI,CAK, NCIC, and KFCB can use the system to monitor online discourse, reduce incitement, and promote national security and cohesion.

5. Social Media Platforms
Companies like Meta (Facebook/Instagram), TikTok Kenya, and Twitter/X can enhance their content moderation by integrating locally trained AI models.

6. Content Moderators
Moderation teams in firms such as Sama will benefit from faster and more accurate identification of toxic content, reducing manual workload.

7. AI and Machine Learning Experts
Kenyan developers, researchers, and data scientists can build and improve models tailored to local languages and cultural nuances.

8. Advocacy and Human Rights Groups
Organizations like Article 19 EA and Ushahidi can use system insights to support digital rights and safer online engagement.

9. Educational Institutions and Researchers
Universities and think tanks will gain access to real-time data for studying online behavior, digital communication, and policy development.

10. Advertisers and Brands
Companies like Safaricom and Equity Bank will benefit from a cleaner digital environment, avoiding association with harmful or false content.

# Data Understanding.
## Column Description.
This dataset contains multiple fields that provide rich context for each tweet. Below is a breakdown of what each column means:

| Column              | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `id`                | Unique identifier for each tweet                                            |
| `user_name`         | Twitter handle of the user who posted the tweet                            |
| `user_location`     | Self-declared location of the user (may be empty or vague)                  |
| `user_description`  | Bio/description from the user's Twitter profile                            |
| `user_created`      | Date when the user's Twitter account was created                           |
| `user_followers`    | Number of followers the user has                                           |
| `user_friends`      | Number of accounts the user is following                                   |
| `user_favourites`   | Number of tweets the user has liked                                        |
| `user_verified`     | Whether the user account is verified (True/False)                          |
| `date`              | Timestamp of when the tweet was posted                                     |
| `text`              | Full content of the tweet                                                  |
| `hashtags`          | List of hashtags used in the tweet (may be NaN if none were used)          |
| `source`            | Device or app used to send the tweet (e.g., Twitter for Android, Web App)  |
| `retweets`          | Number of times the tweet has been retweeted                               |
| `favorites`         | Number of times the tweet has been liked                                   |
| `is_retweet`        | Whether the tweet is a retweet (True/False)                                |

## Explotary Data Analysis
### Tweet length distribution.
Most tweets range from 120 to 145 characters, clustering near the original tweet character limit.This indicates users tend to write concise but full messages, especially in political contexts.

![image](https://github.com/user-attachments/assets/3d2a21c6-65db-4227-afaa-4988ed3c1d61)

### Tweet Volume over time.
This plot shows how many tweets were posted on each day. Peaks in the plot can reveal peak,rise during important political events, spikes in conversation volume and time periods worth focusing on during analysis

![image](https://github.com/user-attachments/assets/81273b2d-64c3-4622-9ac7-a05ac649d849)

### Top 10 User location
The top user location is Nairobi, Kenya, appearing over 2,900 times, with variations like Nairobi, Nairobi Kenya, and Kenya | Nairobi also common. Kenya alone appears over 1,300 times. These duplicate entries will require standardization during preprocessing, such as simplifying "Nairobi, Kenya" to "Nairobi" and reconsidering whether to keep country-level entries like "Kenya".

![image](https://github.com/user-attachments/assets/a0de4e75-4805-4702-93b2-612c1b84e3e5)

### Most frequent tweets in word cloud 
Tweets frequently mention names and political figures such as Uhuru Kenyatta, Raila Odinga, William Ruto, and Martha Koome. Other top terms include president, deputy, justice, court, government, and bbi. This indicates that conversations are centered around national leadership, governance, and political events, which strongly aligns with our project’s focus.

![image](https://github.com/user-attachments/assets/fb1cc46c-aca1-4071-bd3d-6957f90da973)

### Bigram and Trigram
The top bigrams include uhuru kenyatta, president uhuru, william ruto, raila odinga, and deputy president, while the top trigrams are president uhuru kenyatta, deputy president william, state house kenya, and justice martha koome. This shows that tweets often follow structured formats combining titles and names, reinforcing the relevance of using n-grams and embeddings in modeling.

![image](https://github.com/user-attachments/assets/9a002036-7bde-4e2c-adfd-a377e8c11a4e)

### Top 10 Users by Total Retweets

![image](https://github.com/user-attachments/assets/319f12a7-014d-4ca1-ba62-e0ec73d310c6)


# Data Preparation and Preprocessing.

Missing Values: Removed rows with missing tweets or user IDs; imputed or tagged non-critical fields like location as Unknown.

User Location Normalization: Standardized user locations using lowercase, regex cleaning, and basic geolocation mapping.

Tweet Cleaning: Removed URLs, mentions, hashtags, emojis, and special characters; applied lowercasing, tokenization, stopword removal, and optional lemmatization.

Content & Engagement Analysis: Explored frequent topics, hashtags, and engagement trends (likes/retweets) by sentiment and region.

Bigrams & WordClouds: Extracted common word pairs and visualized frequent terms using word clouds to highlight dominant tweet themes.








