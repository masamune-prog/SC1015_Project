# FDDC Team 5 Fake News Classification

### About

This is a Mini Project for SC1015 (Introduction to Data Science and Artificial Intelligence) which focuses on fake news dectection using the [ISOT News Dataset](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets). For detailed walkthrough, please view the source code in order from

1. [Data Visualisation](https://github.com/masamune-prog/SC1015_Project/blob/edits/Data%20cleaning%20%2B%20visualization.ipynb)
2. [Data Cleaning and Augmentation](https://github.com/masamune-prog/SC1015_Project/blob/edits/Data%20Cleaning%20%2B%20Preprocessing%20%2B%20Feature%20Engineering%20.ipynb)
3. [Decision Tree](https://github.com/masamune-prog/SC1015_Project/blob/edits/Training%20Attempt%20%231%20Using%20Indicators.ipynb)
4. [Logistic regression with TF-IDF](https://github.com/masamune-prog/SC1015_Project/blob/edits/Training%20Attempt%20%232%20Using%20Textual%20Data.ipynb)
5. [Transformer Approach using DistillBERT](https://github.com/masamune-prog/SC1015_Project/blob/edits/Training%20Attempt%20%233%20Using%20Deep%20Learning.ipynb)

### Problem Definition

- Are we able to tell if news is fake or not?
- What is the best model to predict this

### Conclusion

- Sentiment in News Reports is a not good indicator of veracity(surprising!)
- Word Count in Fake News is generally longer(People do not read whole text)
- Deep Learning Approach using DistillBERT consistently performed well in predicting fake news, but took most resources to train(99.9% accuracy, 99.9% recall)
- Producing word embeddings improved accuracy significantly(94% accuracy(TD-IDF) vs 71% accuracy(Decision Tree) vs 99.97%(DistillBERT) )
- Yes, it is possible to predict wherether news is fake or not, but require continous training to ensure model remains current.
- Implementation of a MLops Pipeline may be more long term solution to expand database and train daily

### Evaluation on a uniform test set

| Model                                | Training Accuracy | Test Accuracy |
| ------------------------------------ | ----------------- | ------------- |
| Decision Tree + Random Forests       | 71%               | 70%           |
| Logistic Regression with TF-IDF      | 97%               | 97%           |
| Transformer(Pre-Trained DistillBERT) | 100%              | 99.9%         |

### What did we learn in this project?

- The myth of [class imbalance](https://towardsdatascience.com/your-dataset-is-imbalanced-do-nothing-abf6a0049813)
- Word Embeddings and Transformer Architecture
- HuggingFace Transformers
- Logistic Regression from sklearn
- Collaborating using GitHub
- Concepts about Precision, Recall, and F1 Score

### Contributors

- @Poto5qin Data Cleaning + Preprocessing + Feature Engineering + Decision Tree
- @adimi9 Exploratory Data Analysis + Random Forest + TF-IDF Vectorisation + Logistic Regression
- @masamune-prog Transformer Approach

### References

- https://huggingface.co/docs/transformers/en/index
- https://jon-dagdagan.medium.com/fake-news-detection-pre-processing-text-d9648a2854e5
- https://anderk687.medium.com/building-a-fake-news-classification-app-using-keras-streamlit-32f30afb71ad
- https://prabhitha3.medium.com/fake-news-detection-using-machine-learning-models-4475f62c0836
- https://huggingface.co/docs/transformers/en/model_doc/distilbert
- https://discuss.huggingface.co/t/the-point-of-using-pretrained-model-if-i-dont-freeze-layers/40675
- https://www.researchgate.net/figure/The-DistilBERT-model-architecture-and-components_fig2_358239462
