import streamlit as st
import pickle
import sklearn

model =pickle.load(open('model.pkl', 'rb'))
tv= pickle.load(open('vectorizer.pkl','rb'))

st.title('SMS/Email Spam Classifier')
st.subheader('by Mayank Srivastava', divider='rainbow')
text = st.text_area('Enter your SMS/ email')
if st.button('Predict'):
    import re
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    from nltk.corpus import stopwords

    from nltk.stem import PorterStemmer
    from nltk.stem import WordNetLemmatizer
    import string
    punctuation= string.punctuation
    # stop words
    from nltk.corpus import stopwords
    stops= set(stopwords.words('english'))
    #stem lemma
    stem = PorterStemmer()
    lemm = WordNetLemmatizer()


    def standardize(text):

        # change to lower case
        text = text.lower()

        # keep only alpha numerics
        # assuming _ sign is used for space in text, replacing it with space
        text = re.sub('_', '', text)
        text = re.findall(r"\w+", text)

        # now text has been converted into list of words , after re.findall

        # remove punctuations and stopwords
        text = [i for i in text if i not in stops and i not in punctuation]

        # lemmatization
        text = [lemm.lemmatize(i) for i in text]

        # stemming
        text = [stem.stem(i) for i in text]

        return (' '.join(text))

    # preprocess
    text= standardize(text)

    # vectorize
    text = tv.transform([text])

    # predict
    pred =model.predict(text)
    confidence =str(round(model.predict_proba(text).max()*100,2))+ "% confidence"

    # display
    if pred[0] == 0:
        st.header("Not Spam")
        st.header(confidence)
    else:
        st.header('Spam')
        st.header(confidence)
