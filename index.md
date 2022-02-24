### Introduction/Background

The popular social media site Reddit was founded in 2005 and has become one of the most influential social media platforms in the world. On the site, users can create subreddits, or a forum dedicated to a specific topic. One such subreddit is called “Am I the Asshole?” (AITA), and it allows users to post personal experiences or moral dilemmas so that other users can vote on whether the poster (OP) was in the wrong, or if they were the “asshole.”

### Problem Definition

We aim to use a machine learning based prediction system to determine what verdict a specific post will receive if posted to the AITA subreddit. Such a system can allow users to circumvent the subreddit altogether, which can eliminate hateful or offensive responses that posters are often subject to when voters are passionate about their post. This system can also help us understand how the verbiage chosen to tell a story can affect how people respond to it.

There are tens of thousands of posts on the AITA subreddit that can be accessed using Reddit’s API. We will use a subset of these posts for our predictive model. Each post on AITA has a user, timestamp, title, a varying number of comments, a post body with a character limit of 3,000, and a verdict. 

### Methods

- Word2Vec (Post2Vec):

Word2Vec is a word embedding technique that takes a group of texts and finds correlations between words by putting them into a vector space. Each vector consists of a few hundreds of dimensions, and a vector in the vector space represents a word. The proximity of any two vectors means that the two vectors are highly correlated. Considering the specificity of posts, we are going to split a post into two parts, title and story, and apply the method for both parts.

- BERT (Bidirectional Encoder Representations from Transformers):

Like the Word2Vec Method, BERT learns contextual relations between words. As its name suggests, BERT supports two-way learning models and transfer learning. Also, it is possible to infer semantic and grammatical information between words.

- Naïve Bayesian

We are looking to compare the accuracy and the precision of Word2Vec and BERT and appropriately combine them. Also, utilizing Naïve Bayesian, we are going to factor not only the posts itself, but the length of a post, gender and age into the process so that we can capture the tendencies that might have significantly affected the results.

### Potential Results

We hope to be able to compare our different models and determine which is the most accurate at predicting a post’s judgment. We expect there to be multiple factors that can increase the likelihood of a poster being declared “the asshole,” such as factors used by a Naive Bayesian network such as age or relationships, or factors used in Word2Vec such as the correlation between a title and a post. We expect the results to help us determine which factors are the largest contributor to a post’s verdict.


### Link to Gantt Sheet

Click on the following [link](https://docs.google.com/spreadsheets/d/1c5EcHU4atJxC3LtkHKbG2dMgIPzFvhRl5RgATvWg2Nk/edit?usp=sharing) to see our Gannt Sheet. 

### References 

Mali, A., & Sedamkar, R. Prediction of Depression Using Machine Learning and NLP Approach, International Journal of Intelligent Communication, Accessed February 21, 2022, Retrieved from [https://www.ijiccn.com/images/files/vol2-issue1/Prediction-of-depression-using-Machine-Learng-and-NLP-approach.pdf](https://www.ijiccn.com/images/files/vol2-issue1/Prediction-of-depression-using-Machine-Learng-and-NLP-approach.pdf)

Wang, I. “Am I the Asshole?”: A Deep Learning Approach for Evaluating Moral Scenarios, Stanford University, Accessed February 21, 2022, Retrieved from [http://cs230.stanford.edu/projects_spring_2020/reports/38963762.pdf](http://cs230.stanford.edu/projects_spring_2020/reports/38963762.pdf)

Botzer, N., Gu, S., Weninger, T. (2021) Analysis of Moral Judgment on Reddit. Retrieved from [http://arxiv.org/abs/2101.07664 arXiv:2101.07664](https://arxiv.org/abs/2101.07664)

Sarat, P., Kaundinya, P., Mujumdar, R., Dambekodi, S. (2020) Can Machines Detect if you’re a Jerk?, Retrieved from [https://rohitmujumdar.github.io/projects/aita.pdf](https://rohitmujumdar.github.io/projects/aita.pdf)

Jang, B., Kim, I., & Kim, J. (2019) Word2vec convolutional neural networks for classification of news articles and tweets. PLOS ONE 14(8): e0220976. Retrieved from [https://doi.org/10.1371/journal.pone.0220976](https://doi.org/10.1371/journal.pone.0220976)

O'Brien, E. (2020). iterative/aita_dataset: Praw rescrape of entire dataset (v.20.1.2). Zenodo. Retrieved from [https://doi.org/10.5281/zenodo.3677563](https://doi.org/10.5281/zenodo.3677563)
