### Introduction

The popular social media site Reddit was founded in 2005 and has become one of the most influential social media platforms in the world. On the site, users can create subreddits, or a forum dedicated to a specific topic. One such subreddit is called “Am I the Asshole?” (AITA), and it allows users to post personal experiences or moral dilemmas so that other users can vote on whether the poster (OP) was in the wrong, or if they were the “asshole.”

### Problem Statement

We aim to use a machine learning based prediction system to determine what verdict a specific post will receive if posted to the AITA subreddit. Such a system can allow users to circumvent the subreddit altogether, which can eliminate hateful or offensive responses that posters are often subject to when voters are passionate about their post. This system can also help us understand how the verbiage chosen to tell a story can affect how people respond to it.

There are tens of thousands of posts on the AITA subreddit that can be accessed using Reddit’s API. We will use a subset of these posts for our predictive model. Each post on AITA has a user, timestamp, title, a varying number of comments, a post body with a character limit of 3,000, and a verdict.

### Data Cleaning and Visualization
Using Reddit API, we collected data from 2012, when the subreddit was created, to 2020. Our current dataset contains approximately 1 million posts, and the items each post contains are as follows: timestamp, title, body, verdict(4 different answers), score, the number of comments, and the final result if the post writer was the “asshole.” For training on BERT, the verdicts were grouped together to convert the problem to that of binary classification. This modification fit naturally with the base dataset since within the verdicts, the two which were combined were already subsets of the parent verdict they were combined with. Some rudimentary data cleaning techniques such as normalizing font case and balancing data classes were performed. As our second and third methods are completed, we plan to use the data from 2021 to measure the accuracy and the precision of our methods. The figures below show the results organized by verdict and the number of characters in the title or text. Although no immediate trends appear from these plots, an important takeaway is the much larger amount of “asshole” verdicts that must be accounted for while training the methods. Additionally, the AITA subreddit generally limits posts to a 3000 character limit, which can be seen in Figure 2.

| ![title length vs number of posts](./images/title_length_vs_num_posts.png) |
|:--:|
| Figure 1. Number of Character in Title by Verdict |

| ![body length vs number of posts](./images/body_length_vs_num_posts.png) |
|:--:|
| Figure 2. Number of Character in Text by Verdict |

### Methods

- Word2Vec (Post2Vec):

Word2Vec is a word embedding technique that takes a group of texts and finds correlations between words by putting them into a vector space. Each vector consists of a few hundreds of dimensions, and a vector in the vector space represents a word. The proximity of any two vectors means that the two vectors are highly correlated. Considering the specificity of posts, we are going to split a post into two parts, title and story, and apply the method for both parts.

- BERT (Bidirectional Encoder Representations from Transformers):

Like the Word2Vec Method, BERT learns contextual relations between words. As its name suggests, BERT supports two-way learning models and transfer learning. Also, it is possible to infer semantic and grammatical information between words.

- Naïve Bayesian

We are looking to compare the accuracy and the precision of Word2Vec and BERT and appropriately combine them. Also, utilizing Naïve Bayesian, we are going to factor not only the posts itself, but the length of a post, gender and age into the process so that we can capture the tendencies that might have significantly affected the results.

### Bert Implementation
We utilized the huggingface library in implementing our Bert model. The huggingface library is a NLP library that provides multiple off the shelf models. In this case, we used the “bert-base-uncased” model within the library. We added an additional linear layer after the pre-trained Bert. When training, we only modify the parameters in the last layer of Bert and the parameters in the added linear layer. This way, we can leverage the pretrained parameters containing semantics of the English vocabulary. We fine tuned the model to our specific task of judging whether a post is considered “asshole.”

### Results of Implemented Method
After training a few models with different hyperparameters, we converged on the following values for the hyperparameters of our Bert model:

| Parameter | Value |
| -- | -- |
| Epochs | 4 |
| Beta coefficients | 0.9, 0.999 |
| Weight Decay | 0.01 |
| Batch Size | 8 |
| Max Length | 500 |

Some of the above parameters were default to the Bert implementation itself, while others were modified, either through trial and error or through independent research on supposedly optimal values. It is worth noting that, as will be discussed further, the results of this method were not as good as desired. This could be evidence of the fact that there is not necessarily a discernible pattern based on the input data of the final verdict of a post on the AITA subreddit, the Bert method is not suitable for this problem, or the hyperparameters have room for optimization.

Prior to the final report, we plan to continue to alter the hyperparameters on upcoming iterations of our model to achieve more robust results, and the other methods we plan to implement will ideally achieve better results.

Our final model reached an accuracy of approximately 63.2%. If we define an asshole verdict as a positive result, we achieved a precision of approximately 83.7%, and a recall of 62.4%. Figure 3 and 4 show the normalized and unnormalized confusion matrices from our model.

| ![Figure 3. The confusion matrix for our final Bert-implemented model](./images/confusion_matrix.png) |
|:--:|
| Figure 1. Number of Character in Title by Verdict |

| ![body length vs number of posts](./images/normalised_confusion_matrix.png) |
|:--:|
| Figure 4. The normalized confusion matrix for our final Bert-implemented model |

For this model, it is worth noting that we only considered data points that received either that “asshole” or “not the asshole” verdict—we did not consider the “no assholes here” or the “everyone sucks here” verdicts. These verdicts make up such a small fraction of the data points that we neglected them, however, prior to the final report we may consider these to analyze the effect of adding these verdicts to the Bert model, as well as the models we will implement in the future.

### Link to Gantt Sheet

Click on the following [link](https://docs.google.com/spreadsheets/d/1c5EcHU4atJxC3LtkHKbG2dMgIPzFvhRl5RgATvWg2Nk/edit?usp=sharing) to see our Gannt Sheet.

### References

Mali, A., & Sedamkar, R. Prediction of Depression Using Machine Learning and NLP Approach, International Journal of Intelligent Communication, Accessed February 21, 2022, Retrieved from [https://www.ijiccn.com/images/files/vol2-issue1/Prediction-of-depression-using-Machine-Learng-and-NLP-approach.pdf](https://www.ijiccn.com/images/files/vol2-issue1/Prediction-of-depression-using-Machine-Learng-and-NLP-approach.pdf)

Wang, I. “Am I the Asshole?”: A Deep Learning Approach for Evaluating Moral Scenarios, Stanford University, Accessed February 21, 2022, Retrieved from [http://cs230.stanford.edu/projects_spring_2020/reports/38963762.pdf](http://cs230.stanford.edu/projects_spring_2020/reports/38963762.pdf)

Botzer, N., Gu, S., Weninger, T. (2021) Analysis of Moral Judgment on Reddit. Retrieved from [http://arxiv.org/abs/2101.07664 arXiv:2101.07664](https://arxiv.org/abs/2101.07664)

Sarat, P., Kaundinya, P., Mujumdar, R., Dambekodi, S. (2020) Can Machines Detect if you’re a Jerk?, Retrieved from [https://rohitmujumdar.github.io/projects/aita.pdf](https://rohitmujumdar.github.io/projects/aita.pdf)

Jang, B., Kim, I., & Kim, J. (2019) Word2vec convolutional neural networks for classification of news articles and tweets. PLOS ONE 14(8): e0220976. Retrieved from [https://doi.org/10.1371/journal.pone.0220976](https://doi.org/10.1371/journal.pone.0220976)

O'Brien, E. (2020). iterative/aita_dataset: Praw rescrape of entire dataset (v.20.1.2). Zenodo. Retrieved from [https://doi.org/10.5281/zenodo.3677563](https://doi.org/10.5281/zenodo.3677563)
