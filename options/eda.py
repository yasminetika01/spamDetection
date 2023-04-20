import pandas as pd
import streamlit as st
from io import BytesIO, StringIO
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from utils import transform_text
import nltk
from wordcloud import WordCloud


spam_df = pd.read_csv('data/spam.csv',encoding="ISO-8859-1")
sample_df = spam_df.sample(5)

rows, columns = spam_df.shape
spam_info = spam_df.info(verbose= False)
print(spam_df)

buf = StringIO()
spam_df.info(verbose = True, buf=buf, memory_usage=False)
spam_df_info = buf.getvalue()

spam_df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'] , inplace = True)
spam_df.rename(columns={'v1':'target','v2':'text'} , inplace=True )
cleaned_sample_df = spam_df.sample(5)


encoder = LabelEncoder()
spam_df['target'] = encoder.fit_transform(spam_df['target'])
tranformed_df_head =  spam_df.head()
null_values_count = spam_df.isnull().sum()
duplicate_values_count = spam_df.duplicated().sum()

fig1, ax1 = plt.subplots(figsize=(5, 5))
ax1.axis('equal')
ax1 =  plt.pie(spam_df['target'].value_counts(), labels=['ham','spam'],autopct="%0.2f")

spam_df['num_characters'] = spam_df['text'].apply(len)
spam_df['num_words'] = spam_df['text'].apply(lambda x:len(nltk.word_tokenize(x)))
spam_df['num_sentences'] = spam_df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))
stats_spam_df =  spam_df.head()

#ham
ham_describe = spam_df[spam_df['target'] == 0][['num_characters','num_words','num_sentences']].describe()
#spam
spam_describe = spam_df[spam_df['target'] == 1][['num_characters','num_words','num_sentences']].describe()


char_describe_fig = plt.figure(figsize=(12,6))
sns.histplot(spam_df[spam_df['target'] == 0]['num_characters'])
sns.histplot(spam_df[spam_df['target'] == 1]['num_characters'],color='red')

words_describe_fig = plt.figure(figsize=(12,6))
sns.histplot(spam_df[spam_df['target'] == 0]['num_words'])
sns.histplot(spam_df[spam_df['target'] == 1]['num_words'],color='red')

pairplot = sns.pairplot(spam_df,hue='target')

heatmap_fig = plt.figure(figsize=(12,6))
sns.heatmap(spam_df.corr(),annot=True)


spam_df['transformed_text'] = spam_df['text'].apply(transform_text)
spam_df.to_csv("data/cleaned_transformed_data.csv", columns=["target", "transformed_text"], index=False)

transformed_spam_df = spam_df.head(6)

wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')

spam_tranformed_text_col = spam_df[spam_df['target'] == 1]['transformed_text']
ham_tranformed_text_col = spam_df[spam_df['target'] == 0]['transformed_text']

spam_wc = wc.generate(spam_tranformed_text_col.str.cat(sep=" "))
ham_wc = wc.generate(ham_tranformed_text_col.str.cat(sep=" "))



corpus_list = []
for msg in spam_tranformed_text_col.tolist():
    for word in msg.split():
        corpus_list.append(word)
common_words = pd.DataFrame(Counter(corpus_list).most_common(30))
spam_common_words_plot = plt.figure(figsize=(12,8)) 
plt.xticks(rotation='vertical')
sns.barplot(common_words, x=common_words[0], y=common_words[1])

corpus_list = []
for msg in ham_tranformed_text_col.tolist():
    for word in msg.split():
        corpus_list.append(word)
common_words = pd.DataFrame(Counter(corpus_list).most_common(30))
ham_common_words_plot = plt.figure(figsize=(12,8)) 
plt.xticks(rotation='vertical')
sns.barplot(common_words, x=common_words[0], y=common_words[1])



def app():

    st.title("EDA")
    with st.expander("Data Sampling", expanded=False):
        st.subheader("Sample data used for training")
        st.caption(f"This dataset has: <b style='color:tomato'>{rows}</b> rows and <b style='color:tomato'>{columns}</b> columns.", unsafe_allow_html=True)
        st.dataframe(sample_df, width=1800)

    with st.expander("Data Cleaning", expanded=False):
        st.subheader("1- Data Cleaning")
        st.text(spam_df_info)

        st.text("After dropping empty columns and renaming the relevant one. Our dataframe start looking like this:")
        st.dataframe(cleaned_sample_df)

        st.text("""We tranform our target column from text to cardinal number:
                0 ---> ham (non-spam email)
                1 ---> Spam
        """)
        st.dataframe(tranformed_df_head)

        st.text(f"We check for missing values:")
        st.dataframe(null_values_count)

    with st.expander("Exploratory Data Analysis", expanded=False):
        st.subheader("2- Exploratory Data Analysis")

        st.text("We check the proportion of each email type present in the dataset ")

        buf = BytesIO()
        fig1.savefig(buf, format="png")
        st.image(buf)

        st.text("""After that we drive some statistics about number of sentences, words and characters
                constituting each email """)

        st.dataframe(stats_spam_df)

        col1, col2 = st.columns(2)
        with col1:
            st.caption("Statistics of spam emails")
            st.dataframe(spam_describe)
        with col2:
            st.caption("Statistics of ham emails")
            st.dataframe(ham_describe)

        st.text("Analyze number of characters present in each email")
        buf = BytesIO()
        char_describe_fig.savefig(buf, format="png")
        st.image(buf)


        buf = BytesIO()
        words_describe_fig.savefig(buf, format="png")
        st.image(buf)


        buf = BytesIO()
        pairplot.savefig(buf, format="png")
        st.image(buf)


        buf = BytesIO()
        heatmap_fig.savefig(buf, format="png")
        st.image(buf)

    with st.expander("Data Processing", expanded=False):
        st.subheader("3- Data Processing")
        st.dataframe(transformed_spam_df)

        col1, col2 = st.columns(2)
        with col1:
            st.caption("Spam World Cloud")
            fig = plt.figure(figsize=(15,6))
            plt.imshow(spam_wc)
            st.pyplot(fig)

        with col2:
            st.caption("ham World Cloud")
            fig = plt.figure(figsize=(15,6))
            plt.imshow(ham_wc)
            st.pyplot(fig)
        

        buf = BytesIO() 
        spam_common_words_plot.savefig(buf, format="png")
        st.image(buf)

        buf = BytesIO()
        ham_common_words_plot.savefig(buf, format="png")
        st.image(buf)
