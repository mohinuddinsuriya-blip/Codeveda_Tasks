"""
Level 3 - Task 3: NLP — Sentiment Analysis
Dataset: Sentiment Dataset
Internship: Codveda Technologies - Data Analysis Intern
Tools: Python, pandas, matplotlib, seaborn (+ manual NLP preprocessing)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter

sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 120

# ─────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────
df = pd.read_csv('3__Sentiment_dataset.csv')
print("=" * 65)
print("     LEVEL 3 - TASK 3: NLP — SENTIMENT ANALYSIS")
print("=" * 65)
print(f"\n📌 Dataset Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nFirst 3 Rows:")
print(df[['Text', 'Sentiment']].head(3))

# ─────────────────────────────────────────────
# 2. CLEAN & MAP SENTIMENTS
# ─────────────────────────────────────────────
print("\n" + "─" * 65)
print("📌 Step 1: Cleaning Data & Mapping Sentiments")
print("─" * 65)

df = df[['Text', 'Sentiment']].dropna()
df['Text']      = df['Text'].astype(str).str.strip()
df['Sentiment'] = df['Sentiment'].astype(str).str.strip()

positive_set = {
    'Positive','Joy','Excitement','Contentment','Gratitude','Happy','Hopeful',
    'Elation','Euphoria','Enthusiasm','Determination','Pride','Inspiration',
    'Empowerment','Inspired','Admiration','Calmness','Compassion','Fulfillment',
    'Love','Amusement','Serenity','Playful','Confidence','Harmony','Overjoyed',
    'Blessed','Optimism','Appreciation','Creativity','Wonder','Adventure',
    'Radiance','Rejuvenation','Coziness','Resilience','Zest','Celebration',
    'Captivation','Tranquility','Happiness','Accomplishment','Satisfaction',
    'Reverence','Enchantment','Awe','Curiosity','Anticipation','Hope',
    'Grateful','Compassionate','Proud','Romance','Kindness','Affection',
    'Adoration','Enjoyment','Kind','Friendship','Success','Amazement',
    'Energy','Charm','Ecstasy','Connection','Triumph','Heartwarming',
    'Solace','Breakthrough','Motivation','Freedom','Exploration'
}
negative_set = {
    'Negative','Despair','Grief','Loneliness','Sad','Frustration','Regret',
    'Melancholy','Bitterness','Frustrated','Betrayal','Hate','Bad','Disgust',
    'Anger','Fear','Sadness','Disappointment','Sorrow','Loss','Heartbreak',
    'Desolation','Isolation','Resentment','Jealousy','Jealous','Envious',
    'Envy','Shame','Anxiety','Helplessness','Intimidation','Devastated',
    'Disappointed','Bitter','Fearful','Overwhelmed','Numbness','Boredom',
    'Suffering','Darkness','Desperation','Exhaustion','Heartache','Pressure',
    'Obstacle','Apprehensive','Dismissive'
}

df['Sentiment_Category'] = df['Sentiment'].apply(
    lambda s: 'Positive' if s in positive_set else ('Negative' if s in negative_set else 'Neutral')
)
print(f"   ✅ Mapped {df['Sentiment'].nunique()} labels → Positive / Negative / Neutral")
print(f"\n   Sentiment Distribution:")
print(df['Sentiment_Category'].value_counts())

# ─────────────────────────────────────────────
# 3. TEXT PREPROCESSING (manual, no nltk)
# ─────────────────────────────────────────────
print("\n" + "─" * 65)
print("📌 Step 2: Text Preprocessing")
print("─" * 65)

STOPWORDS = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','he','him','his','himself','she','her','hers','herself','it',
    'its','itself','they','them','their','theirs','themselves','what','which',
    'who','whom','this','that','these','those','am','is','are','was','were',
    'be','been','being','have','has','had','having','do','does','did','doing',
    'a','an','the','and','but','if','or','because','as','until','while','of',
    'at','by','for','with','about','against','between','into','through',
    'during','before','after','above','below','to','from','up','down','in',
    'out','on','off','over','under','again','further','then','once','here',
    'there','when','where','why','how','all','both','each','few','more',
    'most','other','some','such','no','nor','not','only','own','same','so',
    'than','too','very','s','t','can','will','just','don','should','now',
    've','ll','re','m','d','ain','aren','couldn','didn','doesn','hadn',
    'hasn','haven','isn','mightn','mustn','needn','shan','shouldn','wasn',
    'weren','won','wouldn','got','get','go','going','went','come','came'
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return tokens

df['Tokens']       = df['Text'].apply(clean_text)
df['Cleaned_Text'] = df['Tokens'].apply(lambda x: ' '.join(x))
df['Word_Count']   = df['Tokens'].apply(len)

print("   ✅ Lowercasing applied")
print("   ✅ Special characters removed")
print("   ✅ Tokenization done")
print("   ✅ Stopwords removed (manual list)")
print(f"\n   Sample:")
print(f"   Original : {df['Text'].iloc[0]}")
print(f"   Processed: {df['Cleaned_Text'].iloc[0]}")

# ─────────────────────────────────────────────
# 4. POLARITY SCORING (lexicon-based)
# ─────────────────────────────────────────────
print("\n" + "─" * 65)
print("📌 Step 3: Polarity Scoring (Lexicon-Based)")
print("─" * 65)

POS_WORDS = {
    'beautiful','love','wonderful','great','amazing','happy','joy','good',
    'excellent','fantastic','awesome','enjoy','best','nice','delightful',
    'pleased','glad','positive','excited','fun','like','brilliant','super',
    'perfect','lovely','cheerful','thrilled','lucky','grateful','blessed'
}
NEG_WORDS = {
    'terrible','hate','awful','bad','horrible','sad','angry','worst','ugly',
    'disgusting','annoying','disappointing','depressed','unhappy','frustrated',
    'miserable','dreadful','painful','boring','fear','stress','worry','anxious',
    'negative','poor','dull','upset','nervous','scared','tired','lonely'
}

def polarity_score(tokens):
    pos = sum(1 for t in tokens if t in POS_WORDS)
    neg = sum(1 for t in tokens if t in NEG_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total

def predict_sentiment(score):
    if score > 0:
        return 'Positive'
    elif score < 0:
        return 'Negative'
    else:
        return 'Neutral'

df['Polarity']           = df['Tokens'].apply(polarity_score)
df['Predicted_Sentiment'] = df['Polarity'].apply(predict_sentiment)

print(f"   ✅ Polarity scoring done using positive/negative lexicon")
print(f"\n   Predicted Sentiment Distribution:")
print(df['Predicted_Sentiment'].value_counts())
print(f"\n   Average Polarity: {df['Polarity'].mean():.4f}")

# ─────────────────────────────────────────────
# 5. VISUALIZATION 1 — Sentiment Distribution
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Sentiment Analysis — Distribution', fontsize=15, fontweight='bold')

colors_map = {'Positive': '#4CAF50', 'Negative': '#E91E63', 'Neutral': '#2196F3'}

for ax, col, title in zip(axes,
    ['Sentiment_Category', 'Predicted_Sentiment'],
    ['Original Sentiment Labels (Mapped)', 'Lexicon-Based Predicted Sentiment']):
    counts = df[col].value_counts()
    bars = ax.bar(counts.index, counts.values,
                  color=[colors_map.get(c, '#FF9800') for c in counts.index],
                  edgecolor='white', alpha=0.85)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                str(int(bar.get_height())), ha='center', fontweight='bold')
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')

plt.tight_layout()
plt.savefig('nlp_sentiment_distribution.png', bbox_inches='tight')
plt.show()
print("\n   ✅ Saved: nlp_sentiment_distribution.png")

# ─────────────────────────────────────────────
# 6. VISUALIZATION 2 — Polarity Distribution
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Polarity & Word Count Analysis', fontsize=15, fontweight='bold')

axes[0].hist(df['Polarity'], bins=25, color='#2196F3', edgecolor='white', alpha=0.85)
axes[0].axvline(x=0, color='#E91E63', lw=2, linestyle='--', label='Neutral (0)')
axes[0].axvline(x=df['Polarity'].mean(), color='#4CAF50', lw=2,
                linestyle='--', label=f'Mean ({df["Polarity"].mean():.2f})')
axes[0].set_title('Polarity Score Distribution', fontsize=13)
axes[0].set_xlabel('Polarity (-1 Negative → +1 Positive)')
axes[0].set_ylabel('Frequency')
axes[0].legend()

sns.boxplot(data=df, x='Sentiment_Category', y='Word_Count',
            palette=colors_map, ax=axes[1])
axes[1].set_title('Word Count by Sentiment Category', fontsize=13)
axes[1].set_xlabel('Sentiment')
axes[1].set_ylabel('Word Count (after stopword removal)')

plt.tight_layout()
plt.savefig('nlp_polarity_wordcount.png', bbox_inches='tight')
plt.show()
print("   ✅ Saved: nlp_polarity_wordcount.png")

# ─────────────────────────────────────────────
# 7. VISUALIZATION 3 — Top Words per Sentiment
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Top 15 Words per Sentiment Category', fontsize=15, fontweight='bold')

for ax, sentiment in zip(axes, ['Positive', 'Negative', 'Neutral']):
    tokens = []
    for toks in df[df['Sentiment_Category'] == sentiment]['Tokens']:
        tokens.extend(toks)
    freq = Counter(tokens).most_common(15)
    if freq:
        words, counts = zip(*freq)
        ax.barh(words, counts, color=colors_map[sentiment], edgecolor='white', alpha=0.85)
        ax.invert_yaxis()
    ax.set_title(f'{sentiment} Sentiment', fontsize=13)
    ax.set_xlabel('Frequency')

plt.tight_layout()
plt.savefig('nlp_top_words.png', bbox_inches='tight')
plt.show()
print("   ✅ Saved: nlp_top_words.png")

# ─────────────────────────────────────────────
# 8. VISUALIZATION 4 — Platform & Sentiment
# ─────────────────────────────────────────────
df_full = pd.read_csv('3__Sentiment_dataset.csv')
df_full['Sentiment'] = df_full['Sentiment'].astype(str).str.strip()
df_full['Sentiment_Category'] = df_full['Sentiment'].apply(
    lambda s: 'Positive' if s in positive_set else ('Negative' if s in negative_set else 'Neutral')
)
df_full['Platform'] = df_full['Platform'].astype(str).str.strip()

platform_sentiment = df_full.groupby(['Platform', 'Sentiment_Category']).size().unstack(fill_value=0)

plt.figure(figsize=(10, 6))
platform_sentiment.plot(kind='bar', color=['#E91E63','#2196F3','#4CAF50'],
                        edgecolor='white', alpha=0.85, figsize=(10,6))
plt.title('Sentiment Distribution by Platform', fontsize=15, fontweight='bold')
plt.xlabel('Platform')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Sentiment')
plt.tight_layout()
plt.savefig('nlp_platform_sentiment.png', bbox_inches='tight')
plt.show()
print("   ✅ Saved: nlp_platform_sentiment.png")

# ─────────────────────────────────────────────
# 9. SAVE RESULTS
# ─────────────────────────────────────────────
df[['Text','Sentiment','Sentiment_Category',
    'Cleaned_Text','Polarity','Predicted_Sentiment','Word_Count']].to_csv(
    'sentiment_results.csv', index=False)
print("   ✅ Saved: sentiment_results.csv")

print("\n" + "=" * 65)
print("✅ NLP Sentiment Analysis Completed!")
print(f"   Total Records : {len(df)}")
print(f"   Avg Polarity  : {df['Polarity'].mean():.4f}")
print("   Output Files  : nlp_sentiment_distribution.png")
print("                   nlp_polarity_wordcount.png")
print("                   nlp_top_words.png")
print("                   nlp_platform_sentiment.png")
print("                   sentiment_results.csv")
print("=" * 65)
