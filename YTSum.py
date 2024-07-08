from youtube_transcript_api import YouTubeTranscriptApi
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import transformers
from transformers import BartTokenizer, BartForConditionalGeneration


#www.youtube.com/watch?v=8BBzoOB7nxs

link = input("YouTube Summariser v1.0\nEnter a URL: ")
unique_id = link.split("=")[-1]
trans = YouTubeTranscriptApi.get_transcript(unique_id)
transcript = " ".join([item['text'] for item in trans])

#print(transcript)
#print("\n")

sentences = sent_tokenize(transcript)
organized_sent = {k: v for v, k in enumerate(sentences)}

tf_idf = TfidfVectorizer(
    min_df = 2,
    strip_accents='unicode', 
    max_features=None,
    lowercase=True,
    token_pattern=r'\w{1,}',
    ngram_range=(1, 3),
    use_idf=True,
    smooth_idf=True,
    sublinear_tf=True,
    stop_words='english')

sentence_vectors = tf_idf.fit_transform(sentences)
sent_scores = np.array(sentence_vectors.sum(axis=1)).ravel()

N=6;
top_n_sentences = [sentences[index] for index in np.argsort(sent_scores, axis=0)[::-1][:N]]

mapped_sentences = [(sentence,organized_sent[sentence]) for sentence in top_n_sentences]
mapped_sentences = sorted(mapped_sentences, key = lambda x: x[1])
ordered_sentences = [element[0] for element in mapped_sentences]
summary = " ".join(ordered_sentences)

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

input_tensor = tokenizer.encode(transcript, return_tensors="pt", max_length=512, truncation=True)
outputs_tensor = model.generate(input_tensor, max_length=160, min_length=120, length_penalty=2.0, num_beams=4, early_stopping=True)
print(tokenizer.decode(outputs_tensor[0], skip_special_tokens=True))



