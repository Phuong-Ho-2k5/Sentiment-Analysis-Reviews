import bz2
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

#init stop words and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

#text cleaning function
def text_cleaning(text):
    #lowercase
    text = str(text).lower()
    #remove html tags
    text = re.sub(r'<.*?>', '', text)
    #remove urls
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    #remove non-alphabetic characters
    text = re.sub(r'[^a-z\s]', '', text)
    #tokenize
    words = text.split()
    #remove stop words and lemmatize
    clean_words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(clean_words)

#parse line from dataset
def parse_line(line):
    #line is bytes, decode to string
    line = line.decode('utf-8')
    #split line into label and text
    label = 0 if line.startswith('__label__1') else 1
    #text is after the first space
    text = line.split(' ', 1)[1].strip()
    return text, label

#main processing function
def process_data_to_csv(input_path, output_path, n_samples=10000):
    texts, labels = [], []
    print(f"Bắt đầu xử lý: {input_path}")
    
    with bz2.open(input_path, 'rb') as f:
        for i, line in enumerate(f):
            if n_samples and i >= n_samples:
                break
            
            #parse line
            raw_text, label = parse_line(line)
            #clean text
            cleaned = text_cleaning(raw_text)
            
            if len(cleaned.split()) > 2: 
                texts.append(cleaned)
                labels.append(label)

            if i % 2000 == 0 and i > 0:
                print(f"Đã xử lý {i} mẫu...")

    df = pd.DataFrame({'text': texts, 'label': labels})

    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Xong! File đã lưu tại {output_path}")

if __name__ == "__main__":
    process_data_to_csv('data/raw/train.ft.txt.bz2', 'data/processed/train_clean.csv', n_samples=100000)
    process_data_to_csv('data/raw/test.ft.txt.bz2', 'data/processed/test_clean.csv', n_samples=20000)