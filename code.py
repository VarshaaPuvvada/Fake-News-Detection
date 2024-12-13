!pip install wordcloud

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from wordcloud import WordCloud

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D,Conv1D,MaxPool1D
from sklearn.model_selection import train_test_split
import re

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import gensim

true_dt = pd.read_csv("True.csv", encoding='utf-8', on_bad_lines='skip')


fake_dt=pd.read_csv("Fake.csv")

true_dt.head()

true_dt.columns

fake_dt.columns

true_dt['subject'].value_counts()

sns.countplot(x='subject',data=true_dt)

text=''.join(fake_dt["text"].tolist())

wordcloud=WordCloud(width=1600,height=800,background_color='black').generate(text)
plt.figure(figsize=(10,10))
plt.imshow(wordcloud)

true_dt.sample(5)

unknown_publishers=[]
for index,row in enumerate(true_dt.text.values):
     try:
      record=row.split(' - ', maxsplit=1)
      record[1]

      assert((len(record[0])<260))
     except:
      unknown_publishers.append(index)


len(unknown_publishers)

true_dt = true_dt.drop(8970, axis=0)


 true_dt.iloc[unknown_publishers].text

publishers=[]
tmp_txt=[]
for index,row in enumerate(true_dt.text.values):
   if index in unknown_publishers:
      tmp_txt.append(row)
      publishers.append('Unknown')
   else:
      record=row.split(' - ',maxsplit=1)
      # Check if the record has been split into two parts
      if len(record) > 1:
          publishers.append(record[0].strip())
          tmp_txt.append(record[1].strip())
      else:
          # Handle the case where there is no ' - ' in the row
          publishers.append('Unknown') # or any other default value
          tmp_txt.append(row)

true_dt['publishers']=publishers
true_dt['text']=tmp_txt

true_dt.shape

fake_dt.text.tolist()

empty_fake_index=[index for index,text in enumerate(fake_dt.text.tolist()) if str(text).strip()==""]

fake_dt.iloc[empty_fake_index]

true_dt["text"]=true_dt["text"]+" "+true_dt["title"]
fake_dt["text"]=fake_dt["text"]+" "+fake_dt["title"]

true_dt["text"].apply(lambda x: str(x).lower())
fake_dt["text"].apply(lambda x: str(x).lower())

true_dt

import nltk
nltk.download('stopwords')  # Download stopwords
nltk.download('punkt')      # Download punkt for word_tokenize
nltk.download('wordnet')    # Download wordnet for lemmatizer

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
import string

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word.lower() not in stop_words]
    text = ' '.join(words)
    return text


true_dt["text"]

true_dt["class"]=1
fake_dt["class"]=0

real=true_dt[["text","class"]]
fake=fake_dt[["text","class"]]

print(real)

data = pd.concat([real, fake], ignore_index=True)

data.sample(10)

!pip install googletrans==4.0.0-rc1

!pip install spacy==2.2.3
!python -m spacy download en_core_web_sm
!pip install beautifulsoup4==4.9.1
!pip install textblob==0.15.3
!pip install git+https://github.com/laxmimerit/preprocess_kgptalkie.git --upgrade --force-reinstall

import preprocess_kgptalkie as ps

data["text"]=data["text"].apply(lambda x: ps.remove_special_chars(x))

import gensim

y=data["class"].values

x=[d.split() for d in data["text"].tolist()]

dim=100
w2v_model=gensim.models.Word2Vec(sentences=x,vector_size=dim,window=10,min_count=1)

len(w2v_model.wv.key_to_index)


tokenizer=Tokenizer()
tokenizer.fit_on_texts(x)

x=tokenizer.texts_to_sequences(x)

tokenizer.word_index

max_len=1000
x=pad_sequences(x,maxlen=max_len)

vocab_size=len(tokenizer.word_index)+1
vocab=tokenizer.word_index

import numpy as np

def get_weight(model):
    # Retrieve vocabulary and vocabulary size from the model
    vocab = model.wv.key_to_index
    vocab_size = len(vocab)

    # Initialize weight matrix with the correct shape
    weight_matrix = np.zeros((vocab_size, model.vector_size))

    # Populate weight matrix with the embeddings
    for word, i in vocab.items():
        weight_matrix[i] = model.wv[word]

    return weight_matrix

# Get the embedding matrix for the model
embedding_vector = get_weight(w2v_model)


# Redefine vocab_size based on the shape of embedding_vector
vocab_size, embedding_dim = embedding_vector.shape

# Define the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_vector], trainable=False))
model.add(LSTM(128, dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model.summary()

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model.fit(x_train,y_train,validation_split=0.3,epochs=6)

y_pred=(model.predict(x_test) >=0.5) . astype(int)

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
accuracy_score(y_test,y_pred)

print(classification_report(y_test,y_pred))

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Step 1: Modify the LSTM Model to Extract Features
input_dim = vocab_size  # Number of unique words in the vocabulary
embedding_dim = embedding_dim  # Same as from your previous embedding matrix
sequence_length = max_len  # Max sequence length used for padding

# Input layer
inputs = Input(shape=(sequence_length,))
embedding_layer = Embedding(input_dim=input_dim, output_dim=embedding_dim, weights=[embedding_vector], trainable=False)(inputs)
lstm_output = LSTM(128, return_sequences=False)(embedding_layer)  # Set return_sequences=False to get a flat output

# Step 2: Create a feature extraction model
feature_extractor = Model(inputs=inputs, outputs=lstm_output)


assert len(x) == len(y), "Mismatch in number of samples between x and y"

# Prepare the data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Extract features using the LSTM feature extractor
train_features = feature_extractor.predict(X_train)
test_features = feature_extractor.predict(X_test)

# Step 3: Train a Random Forest Classifier on Extracted Features
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(train_features, y_train)

# Step 4: Make Predictions and Evaluate the Model
y_pred = rf_classifier.predict(test_features)
accuracy = accuracy_score(y_test, y_pred)

print("Random Forest with LSTM Features Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))


# Example text input
x = ["Broadcom finalized its $69 billion acquisition of VMware in late 2023, marking one of the tech industry's largest acquisitions. Since then, Broadcom has restructured VMware significantly, reducing its product offerings, cutting more than 2,000 jobs, and reorganizing its partnerships"]
# Convert text to sequences using the tokenizer
x_seq = tokenizer.texts_to_sequences(x)

# Pad the sequences to ensure uniform length
x_padded = pad_sequences(x_seq, maxlen=max_len)

# Make the prediction and threshold it at 0.5
prediction = model.predict(x_padded)
