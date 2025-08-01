# Amazon Fine Food Reviews Classification

This project analyzes and classifies Amazon Fine Food Reviews using various machine learning and NLP techniques. The dataset contains user reviews for food products on Amazon, and the goal is to predict the sentiment (Positive/Negative) of each review.

## Project Structure

- `Classification_of_Amazon_Fine_Food_Reviews.ipynb`: Main Jupyter notebook for data cleaning, preprocessing, feature engineering, model training, and evaluation.
- `Classification_using_BERT.ipynb`: Notebook for classification using BERT-based models.
- `Reviews.csv`: Dataset containing Amazon Fine Food Reviews.
- `README.md`: Project documentation.

## Features

- Data cleaning and preprocessing (handling nulls, duplicates, noise).
- Text normalization (stopword removal, lemmatization, punctuation cleaning).
- Feature extraction using Bag of Words and TF-IDF.
- Handling class imbalance with SMOTE and other resampling techniques.
- Model training and evaluation using Logistic Regression, Naive Bayes, Random Forest, and BERT.
- Visualization of results (word clouds, confusion matrices).

## Requirements

- Python 3.x
- Jupyter Notebook
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- nltk
- imbalanced-learn
- wordcloud
- tqdm
- gensim

Install dependencies using pip:

```sh
pip install pandas numpy matplotlib seaborn scikit-learn nltk imbalanced-learn wordcloud tqdm gensim
```

## Usage

1. Download or clone this repository.
2. Place `Reviews.csv` in the project directory.
3. Open `Classification_of_Amazon_Fine_Food_Reviews.ipynb` in Jupyter Notebook.
4. Run the notebook cells to reproduce the analysis and results.

## Dataset

The dataset is from [Kaggle: Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews).

## Results

- Achieved high accuracy in sentiment classification using Logistic Regression with TF-IDF features.
- Explored the impact of class balancing and different feature extraction methods.

## License

This project is for educational