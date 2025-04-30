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
## Column Files
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









