# End-to-End Healthcare Chatbot

mport pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = '../data/healthcare_faq.csv'
df = pd.read_csv(DATA_PATH)
df.head()

# EDA
print('Rows:', len(df))
print('Topics:', df['topic'].nunique())
print('Categories:', df['category'].nunique())
df['topic'].value_counts().head(10)
df['risk_level'].value_counts().plot(kind='bar')
plt.xlabel('risk_level')
plt.ylabel('count')
plt.title('Risk Level Distribution')
plt.show()

#Build TF‑IDF Retriever
corpus = (df['topic'].fillna('') + ' ' + df['question'].fillna('') + ' ' + df['tags'].fillna('')).tolist()
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=50000)
X = vectorizer.fit_transform(corpus)
X.shape

# Retrieval Function (Top-K)
def retrieve(query, top_k=3):
    q = vectorizer.transform([query])
    sims = cosine_similarity(q, X)[0]
    idx = np.argsort(sims)[::-1][:top_k]
    out = df.iloc[idx][['topic','question','answer','category','risk_level']].copy()
    out['score'] = sims[idx]
    return out

retrieve('heartburn at night', top_k=3)

# Simple Chat Response + Safety Checks
EMERGENCY_KEYWORDS = ['chest pain','difficulty breathing','can\'t breathe','shortness of breath','blue lips','fainting','stroke','confusion','seizure','anaphylaxis','self-harm','suicidal']

def is_emergency(text):
    t = (text or '').lower()
    return any(k in t for k in EMERGENCY_KEYWORDS)


def chatbot_reply(user_message, top_k=3):
    if is_emergency(user_message):
        return ' This may be urgent. Please seek immediate help now. If you are in the U.S., call or text 988 for crisis support, or call 911 for emergencies.'
    hits = retrieve(user_message, top_k=top_k)
    best = hits.iloc[0]
    if best['score'] < 0.10:
        return 'I’m not fully sure. Could you share more details (duration, key symptoms, age group, and any red flags like chest pain or breathing difficulty)?\n\n' + DISCLAIMER
    return f"**{best['topic']}**\n\n{best['answer']}\n\n{DISCLAIMER}"

print(chatbot_reply('I get heartburn at night. what should I do?', top_k=3))

# Try  Query
user_query = 'symptoms of low blood sugar'
print(chatbot_reply(user_query, top_k=3))