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

# Feature Engineering & Vectorization

We prepare our features for machine learning models by:
- Merging cleaned tweets with realistic labeled tweets
- Ensuring no missing values in clean text
- Vectorizing text using TF-IDF (Term Frequency-Inverse Document Frequency)
- Splitting data into training and testing sets

# Modeling
We trained the following models:

**1.Logistic Regression** ( baseline model)
For binary classification, we first implemented Logistic Regression, a simple yet effective linear model known for its interpretability and ease of implementation. It served as a strong baseline and was trained on X_train and y_train, then evaluated on X_test.

**2.Support Vector Machine (SVM)** (effective in high-dimensional text spaces)
Next, we applied a Support Vector Machine (SVM) using the LinearSVC model, particularly suitable for high-dimensional data such as TF-IDF vectors. The SVM was trained with a balanced class weight to address any class imbalance in the dataset. Predictions were made on the test set for evaluation

# **Model Evaluation**
## **Evaluation Metrics:**
1.Accuracy – Overall correctness of predictions.

2.Precision – Proportion of predicted hate tweets that were actually hate.

3.Recall – Proportion of actual hate tweets that were correctly identified.

4.F1-Score – Harmonic mean of precision and recall.

5.Confusion Matrix – Visual summary of prediction performance.

6.ROC Curve and AUC Score – Evaluate the model's ability to distinguish between classes.

## **Models Evaluated:**

### 1.Logistic Regression Evaluation
Logistic Regression is a linear model and was used as a baseline for binary classification.

![image](https://github.com/user-attachments/assets/7990f7c7-022c-4e7e-acfe-b34fb02147a6)

Results:
- Accuracy: 67.80%
- Precision: 19.22%
- Recall: 39.23%
- F1 Score: 25.79%

### 2.Support Vector Machine (SVM) Evaluation
SVM is well-suited for high-dimensional text data and was evaluated using the same metrics.

 ![image](https://github.com/user-attachments/assets/4f8bb8a8-f4a4-4c29-bf50-274b3d53f749)

 Results:

- Accuracy: 68.74%
- Precision: 19.52%
- Recall: 38.12%
- F1 Score: 25.82%
  
#### ROC Curve and AUC Score
The ROC (Receiver Operating Characteristic) curve visualizes the trade-off between True Positive Rate (Recall) and False Positive Rate at various threshold settings.The AUC (Area Under the Curve) summarizes this curve in a single number:

AUC = 1.0 → Perfect classifier

AUC = 0.5 → No discrimination (random guessing)

Higher AUC → Better overall model performance

#### ROC-AUC Interpretation

- **Logistic Regression AUC**: 0.5882
- **SVM AUC**: 0.5743

Both models perform slightly better than random guessing (AUC = 0.5), but still far from ideal.Logistic Regression has a slightly higher AUC than SVM in this task.This suggests that Logistic Regression has marginally better discrimination ability between hate and safe tweets.

## **Summary of Findings:**

- Both Logistic Regression and SVM perform similarly across all metrics.
- Logistic Regression slightly outperforms SVM in AUC score and Recall.
- However, **both models show low Precision, Recall, and F1-Scores**, meaning that they are **not sufficient for production-grade deployment**.

### **Reference to Production Threshold:**

According to standard production deployment thresholds in machine learning systems (especially sensitive domains like hate speech detection);
- Accuracy should ideally exceed **80%**  
- F1-Score should be **at least 50% or higher**  
- Precision and Recall must be balanced to avoid bias

Both Logistic Regression and SVM **fall below these expectations**.

### **Conclusion for Advanced Modeling:**

Given the current outcomes, it is necessary to move to **more advanced models** that can:
- Capture non-linear relationships
- Handle feature interactions more effectively
- Better discriminate subtle patterns in text data

Thus, the next modeling stage will involve:
- **Random Forest Classifier**
- **Multinomial Naive Bayes**
- **XGBoost Classifier**

These models are expected to provide **higher Accuracy, Precision, Recall, and F1-Scores**, aligning better with real-world production deployment standards.

# Advanced Modeling

Given that Logistic Regression and SVM did not meet production-grade performance thresholds,  
we now explore more powerful machine learning models that can better handle the complexity of text classification.

Models used:
- Random Forest Classifier
- Multinomial Naive Bayes
- XGBoost Classifier

Each model will be evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC Curve and AUC Score

These models are more capable of capturing complex patterns, non-linear interactions, and feature importance.


## Improved TF-IDF Vectorization

We use:
- 1-gram and 2-gram combinations (single words and word pairs)
- Max 3000 features

This captures richer language patterns like "uhuru kenyatta", "raila odinga", etc.

### Random Forest Classifier

Random Forest is an ensemble learning method that builds multiple decision trees and merges their predictions to improve accuracy and prevent overfitting.

Advantages:
- Handles feature interactions naturally
- Robust against noise
- Performs well even with limited data cleaning

We will train a Random Forest with 100 trees, and use class weighting to address class imbalance.



### Multinomial Naive Bayes

Naive Bayes is a probabilistic classifier that is particularly effective for text classification problems like spam detection or hate speech detection.

Advantages:
- Very fast and efficient
- Works well with TF-IDF features
- Good for small and medium-sized datasets

We will train a Multinomial Naive Bayes model on our TF-IDF vectorized tweets.


### XGBoost Classifier

XGBoost (Extreme Gradient Boosting) is an advanced boosting algorithm known for its high performance in classification problems.

Advantages:
- High accuracy
- Handles missing values and sparse data
- Efficient computation
- Widely used in competitions and production systems

We will train an XGBoost Classifier without label encoding issues and set `eval_metric='logloss'`.



### Evaluation of Advanced Models

In this section, we evaluate the performance of:
- Random Forest Classifier
- Multinomial Naive Bayes
- XGBoost Classifier

We assess models based on:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC Curve and AUC Score




### Random Forest Evaluation

**Metrics:**
- Accuracy: 83.76%
- Precision: 32.64%
- Recall: 12.98%
- F1-Score: 18.58%
- AUC Score: 0.5951

![image](https://github.com/user-attachments/assets/dec61115-2e3f-4940-bf30-40ea86e5046b)


**Confusion Matrix Insights:**
- 47 hate tweets correctly classified
- 315 hate tweets missed (false negatives)
- Some false positives (97 safe tweets misclassified as hate)

Random Forest improves accuracy compared to Logistic Regression and SVM but still struggles with hate speech detection recall.


## Multinomial Naive Bayes Evaluation

**Metrics:**
- Accuracy: 86.56%
- Precision: 74.42%
- Recall: 8.84%
- F1-Score: 15.80%
- AUC Score: 0.5875

![image](https://github.com/user-attachments/assets/24451f05-4a76-4c6d-9b4e-6fbd863b21c2)


**Confusion Matrix Insights:**
- Very few hate tweets correctly classified (only 32)
- Extremely low recall
- High precision means that when it predicts hate, it is usually correct.

Naive Bayes tends to be conservative, predicting fewer hate cases but with higher precision.


###  XGBoost Classifier Evaluation

**Metrics:**
- Accuracy: 86.76%
- Precision: 84.21%
- Recall: 8.84%
- F1-Score: 16.00%
- AUC Score: 0.5805

![image](https://github.com/user-attachments/assets/98b7bb82-c484-489c-b39b-2f25c2827d42)


**Confusion Matrix Insights:**
- Only 32 hate tweets correctly identified
- Very low recall but extremely high precision
- Slightly better balance compared to Naive Bayes

XGBoost achieves the highest precision (84.21%), crucial for production systems aiming to minimize false accusations.


### ROC Curve and AUC Score for All Models

We plot ROC Curves for all models and calculate their AUC (Area Under Curve) scores.

- Higher AUC → better model distinguishing between hate and safe tweets
- AUC = 0.5 → random guess
- AUC = 1.0 → perfect classifier

<img width="337" alt="ROC interpretation" src="https://github.com/user-attachments/assets/ebf834a3-3969-4c94-9c1c-e2dd50eca935" />





All models perform slightly better than random guessing (AUC = 0.5).

Random Forest achieves the highest AUC (0.5951) among the three.

ROC curves show that none of the models are perfect, but improvements are visible compared to simple baselines like Logistic Regression and SVM.




## Model Recommendation for Deployment Based on Final Metrics

After evaluating all models across key metrics (Accuracy, Precision, Recall, F1-Score, AUC), here are the summarized results:

<img width="356" alt="model recommendation" src="https://github.com/user-attachments/assets/e5ed55f5-61d2-4089-bffc-2410c53714bc" />


### Interpretation:

- **Accuracy**:  
  - XGBoost (86.76%) and Naive Bayes (86.56%) perform best.
- **Precision**:  
  - XGBoost achieves the highest precision (84.21%), meaning when it predicts hate speech, it is often correct.
- **Recall**:  
  - All models show low recall (~8.84% - 12.98%), indicating difficulty in catching hate speech, possibly due to imbalanced data.
- **F1-Score**:  
  - Random Forest has slightly higher F1-score (18.58%) than others but with much lower precision.
- **AUC Score**:  
  - Random Forest has the highest AUC (0.5951), indicating slightly better overall separability.



### Final Recommendation:

Considering all metrics:

<img width="246" alt="Final Recommendation" src="https://github.com/user-attachments/assets/95204d1e-2265-4921-8e8c-ec311abafad5" />


Therefore, **we recommend deploying the XGBoost Classifier** for hate speech detection.


### Key Justification for Choosing XGBoost:

- It achieves **the highest Precision (84.21%)**, critical for minimizing false accusations of hate speech.
- It maintains **the highest Accuracy (86.76%)** across the dataset.
- It offers strong performance even with imbalanced data.
- XGBoost is efficient, scalable, and widely used in production environments.

---

We will save the trained XGBoost model and prepare for Deployment using Streamlit app.


#  Misinformation Detection (Fact-Check Matching)

In this section, we extend the functionality of our hate speech detection system by introducing **misinformation detection**.

The goal is to match incoming tweets against a database of **known false claims** from trusted fact-checking organizations such as PesaCheck and AfricaCheck.

If a tweet is highly similar to a known false claim, we flag it as potential misinformation.

We use a Natural Language Processing (NLP) approach combining:
- TF-IDF vectorization
- Cosine similarity measurement

## Load Known Fact-Checked Claims

We start by loading a set of manually entered false claims based on realistic Kenyan political misinformation.

Each claim includes:
- The false claim text
- The verdict (all False for this exercise)
- The source (PesaCheck, AfricaCheck)

In the future, this can be expanded by scraping actual claims from trusted websites.



## Preprocess Fact-Check Claims

We apply standard text cleaning techniques to the claims to ensure consistency:
- Lowercasing
- Removing URLs, punctuation, special characters
- Removing English stopwords

This ensures that the TF-IDF vectorization later is meaningful.


##  TF-IDF Vectorization

We vectorize:
- All cleaned tweet texts
- All cleaned fact-check claims

This converts text into numerical representations based on term frequency-inverse document frequency scores, preparing them for cosine similarity comparison.



## Compute Cosine Similarity

For each tweet, we compute the cosine similarity with each known false claim.

- A similarity score of 1.0 means identical text
- A similarity score of 0.0 means completely different text

This allows us to measure how close a tweet is to a known piece of misinformation.


## Flagging Tweets as Misinformation
To identify potential misinformation, we apply a similarity threshold based on comparisons with known false claims.
Flagging Criteria:
If a tweet has a similarity score ≥ 0.85 with any known false claim,
It is flagged as potential misinformation.
A new binary column misinformation_flag is added to the dataset:

-True → Tweet is flagged as misinformation

-False → Tweet is not flagged

This step helps isolate content that closely resembles verified false information, supporting future filtering, moderation, or reporting strategies.

###  Summary of our Misinformation Detection

- We loaded 5 manually entered false claims from trusted fact-check organizations.
- Cleaned the text for both tweets and claims.
- Vectorized text using TF-IDF and computed cosine similarity.
- Flagged tweets with similarity ≥ 85% to known false claims.
- Identified tweets that could potentially mislead the public.

This misinformation detection module can be expanded in the future by scraping real claims from sources like PesaCheck and AfricaCheck, updating the detection system dynamically.

This functionality significantly strengthens the practical impact of the hate speech detection system.









