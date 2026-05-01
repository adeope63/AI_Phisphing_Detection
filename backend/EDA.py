import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer


CURRENT_FILE = __file__.replace('\\', '/')
BACKEND_FOLDER = CURRENT_FILE.rsplit('/', 1)[0]
DATASET_FILE = BACKEND_FOLDER + '/CEAS_08.csv'
OUTPUT_FOLDER = BACKEND_FOLDER + '/output'


# Load dataset
data = pd.read_csv(DATASET_FILE)

# Clean columns
data['label'] = data['label'].astype(int)
data['sender'] = data['sender'].fillna('').astype(str)
data['receiver'] = data['receiver'].fillna('').astype(str)
data['date'] = data['date'].fillna('').astype(str)
data['subject'] = data['subject'].fillna('').astype(str)
data['body'] = data['body'].fillna('').astype(str)
data['urls'] = data['urls'].fillna('').astype(str)

# Show basic information
print('\nDataset Shape')
print(data.shape)

print('\nDataset Columns')
print(data.columns.tolist())

print('\nFirst 5 Rows')
print(data.head())

print('\nMissing Values')
print(data.isna().sum())

print('\nLabel Counts')
print(data['label'].value_counts().sort_index())

# Legit vs phishing counts
label_counts = data['label'].value_counts().sort_index()
legit_count = label_counts.get(0, 0)
phishing_count = label_counts.get(1, 0)
total_count = len(data)

legit_ratio = legit_count / total_count
phishing_ratio = phishing_count / total_count

print('\nEmail Class Ratios')
print(f'Legit ratio   : {legit_ratio:.4f}')
print(f'Phishing ratio: {phishing_ratio:.4f}')

# 1. Class distribution
plt.figure(figsize=(7, 5))
plt.bar(['Legit', 'Phishing'], [legit_count, phishing_count])
plt.title('Class Distribution')
plt.xlabel('Email Class')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + '/class_distribution.png')
plt.close()

# 2. Ratio chart
plt.figure(figsize=(7, 5))
plt.bar(['Legit Ratio', 'Phishing Ratio'], [legit_ratio, phishing_ratio])
plt.title('Ratio of Legit and Phishing Emails')
plt.xlabel('Class')
plt.ylabel('Ratio')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + '/label_ratio_bar_chart.png')
plt.close()

# 3. Pie chart
plt.figure(figsize=(7, 7))
plt.pie(
    [legit_count, phishing_count],
    labels=['Legit', 'Phishing'],
    autopct='%1.1f%%',
    startangle=90
)
plt.title('Label Distribution Pie Chart')
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + '/label_pie_chart.png')
plt.close()

# 4. Email length histogram
body_lengths = data['body'].apply(len)

plt.figure(figsize=(8, 5))
plt.hist(body_lengths, bins=30)
plt.title('Email Body Length Distribution')
plt.xlabel('Body Length')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + '/email_length_histogram.png')
plt.close()

# 5. Missing values
missing_counts = data.isna().sum()

plt.figure(figsize=(8, 5))
plt.bar(missing_counts.index, missing_counts.values)
plt.title('Missing Values by Column')
plt.xlabel('Columns')
plt.ylabel('Missing Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + '/missing_values.png')
plt.close()

# 6. Top subject words
vectorizer = CountVectorizer(stop_words='english', max_features=10)
word_matrix = vectorizer.fit_transform(data['subject'])
word_counts = word_matrix.toarray().sum(axis=0)
words = vectorizer.get_feature_names_out()

top_words = pd.DataFrame({
    'word': words,
    'count': word_counts
}).sort_values(by='count', ascending=False)

plt.figure(figsize=(8, 5))
plt.bar(top_words['word'], top_words['count'])
plt.title('Top Subject Words')
plt.xlabel('Word')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + '/top_subject_words.png')
plt.close()

print('\nEDA files saved in output folder.')