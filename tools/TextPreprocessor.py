import torchtext
from torchtext.vocab import GloVe
from keras.models import Sequential
from keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D, Dropout, MaxPooling1D, Flatten
from tensorflow.keras.metrics import Recall, Precision, F1Score
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam, SGD
import nltk
import string
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from torchtext.vocab import GloVe
import numpy as np

text = np.array(citizenship['text'])
labels = np.array(citizenship_full['Citizenship_Label'])

class TextPreprocessor:
    def __init__ (self, data_text, data_labels):
        self.text = data_text
        self.labels = data_labels
        
    def process_text(self, narratives):
        # Tokenize the text
        narratives = [word_tokenize(narrative) for narrative in narratives]
    
        # Remove punctuation, stop words
        translate_table = str.maketrans('', '', string.punctuation)
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
    
        narratives_cleaned = [[lemmatizer.lemmatize(token.translate(translate_table).lower())
                               for token in narrative
                               if not ((token.isalpha() and len(token) == 1) or (token in stop_words))]
                              for narrative in narratives]
    
        all_tokens = [
            token for narrative in narratives_cleaned for token in narrative]
        token_count = Counter(all_tokens)
    
        # create set for the top 100k most frequent word
        top_count_words = set(
            [word for word, _ in token_count.most_common(100000)])
    
        narratives_truncated = [[word for word in tokens if word in top_count_words] for tokens in
                                narratives_cleaned]
    
        max_seq_len = max([len(seq) for seq in narratives_truncated])
        narratives_padded = []
        pad_token_index = 0
        for seq in narratives_truncated:
            padded_seq = seq[:max_seq_len] + ['<PAD>'] * \
                (max_seq_len - len(seq[:max_seq_len]))
            narratives_padded.append(
                [word if word != '<PAD>' else f'<PAD_{pad_token_index}>' for word in padded_seq])
        return narratives_padded, max_seq_len
    def glove_embeddings(selfprocessed_narratives, embedding_dim=100, mode='pretrained'):
        """
        :param mode: 'pretrained' or 'trainable'
        """
    
        # Flatten the list of narratives and get unique tokens
        all_tokens = set(
            [token for narrative in processed_narratives for token in narrative])
    
        if mode == 'pretrained':
            
            cache_dir = './../vector_cache'
            glove = torchtext.vocab.GloVe(name='6B', dim=100, cache=cache_dir)
            # Extract embeddings for the tokens
            embeddings = {word: glove[word]
                          for word in all_tokens if word in glove.stoi}
            return embeddings
        elif mode == 'trainable':
            pass  # to do
        else:
            raise ValueError("Invalid mode. Choose 'pretrained' or 'trainable'.")
    def process_text_with_labels(data):
    processed_texts = []
    labels = []

    for entry in data:
        label, text = entry.split('\t')
        labels.append(label)
        # Remove leading/trailing spaces or newlines
        processed_texts.append(text.strip())

    return processed_texts, labels
    
    
    def get_average_vector(text, embeddings):
        # Get the dimension from the embeddings
        embedding_dim = len(next(iter(embeddings.values())))
    
        text_vectors = []
        for sentence in text:
            sentence_vector = []
            for word in sentence:
                if word != '<PAD>' and word in embeddings:
                    word_vector = embeddings[word]
                    # Ensure all word vectors have the same shape
                    if word_vector.shape == (embedding_dim,):
                        sentence_vector.append(word_vector)
    
            if sentence_vector:
                # Pad vectors to ensure uniform shape within each sentence
                max_length = max(len(vector) for vector in sentence_vector)
                sentence_vector = [np.pad(
                    vector, (0, max_length - len(vector)), mode='constant') for vector in sentence_vector]
                sentence_vector = np.array(sentence_vector)
                avg_vector = np.mean(sentence_vector, axis=0)
                text_vectors.append(avg_vector)
            else:
                # If no valid word vectors found, add a placeholder vector
                text_vectors.append(np.zeros(embedding_dim))
    
        return np.array(text_vectors)
    def get_embedded_batch(text_batch, embedding_dict):
    embedded_batch = []

    embedding_dim = len(embedding_dict[list(embedding_dict.keys())[0]])
    for text in text_batch:

        word_embeddings = []
        embed_len = embedding_dim
        for word in text:
            # print(word)
            embedding = embedding_dict.get(word, np.zeros(embedding_dim))
            embedding = np.array(embedding)
            # if (len(embedding) != embed_len):
            #     print("length not same")
            #     print(len(embedding))
            #     return 0
            # print(len(embedding))
            word_embeddings.append(embedding)

        word_embeddings = np.array(word_embeddings)

        embedded_batch.append(word_embeddings)

        # print(word_embeddings.shape)

    return embedded_batch
