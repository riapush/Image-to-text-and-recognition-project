import pandas as pd
import string
from nltk.corpus import stopwords

def remove_stop_words(data):
    #list where we will append all the words except stopwords
    new_text= []
    #splitting the sentences and iterating over words
    for word in data.split():
        #if the word is a stop word
        if word in english_stop_words:
            #append the empty space
            new_text.append('')
        else:
            #else if it is not the stop word then append the word
            new_text.append(word)
            
    x= new_text[:]
    new_text.clear()
    return " ".join(x)

input_file = 'processed_images.csv' # i ran this code with IMDB dataset also
output_file = 'processed_images_modified.csv'

df = pd.read_csv(input_file)


df['review'] = df['review'].str.replace('\n', ' ').str.replace('"', '')

# transform to lowercase
df['review']= df['review'].str.lower()

# remove punctuation
exclude = set(string.punctuation)
df['review'] = df['review'].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))

#remove stop words
english_stop_words = stopwords.words('english')
df['review']= df['review'].apply(remove_stop_words)


df.to_csv(output_file, index=False)