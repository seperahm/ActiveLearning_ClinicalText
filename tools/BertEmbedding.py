from transformers import BertTokenizer, BertModel
import os
import torch


class BertEmbedding:
    def __init__(self):
        os.environ['HS_DATASETS_OFFLINE'] = '1'
        os.environ['TRANSFORMER_OFFLINE'] = '1'
        
        # print(dataset[-1])
        self.tokenizer = BertTokenizer.from_pretrained('./../../bertbase', local_files_only=True)
        self.model = BertModel.from_pretrained('./../../bertbase', output_hidden_states=True)
        self.data_embed = []
        self.max_para_len = 0
        
        
    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        return self.model
    
    
    def get_sen_embedding(self, sen):
        encoded = self.tokenizer(sen, padding='max_length', truncation=True, max_length=30, return_tensors='pt')
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
        hidden_states = outputs[2]
        bert_embeddings = torch.sum(hidden_states[-1], dim=1)[0]
        return bert_embeddings
        

    def get_para_embedding(self, para, order):
        sen_embeds = []
        for sen in para:
            sen_embeds.append(self.get_sen_embedding(sen))
        cat_sen_embeds = torch.cat(sen_embeds, dim=0)
        self.max_para_len = max(self.max_para_len, len(cat_sen_embeds))
        print(order, 'sen length: ', len(cat_sen_embeds), '    max length: ', self.max_para_len)
        return cat_sen_embeds
        

    def get_data_embedding(self, data):
        para_embeds = []
        k = 0
        for para in data:
            para_embeds.append(np.array(self.get_para_embedding(para, k)))
            k += 1
        pad_para_embeds = [np.pad(para_embed, (0, self.max_para_len - len(para_embed)), constant_values=0) for para_embed in para_embeds]
        self.data_embed = np.array(pad_para_embeds)
        return self.data_embed
            
    
    def print_input_ids(self, test_sen, tokenized_sen, input_ids):
        print('======================', test_sen, '===============================')
        for tup in zip(tokenized_sen, input_ids): 
            print('{:<12} {:>6}'.format(tup[0], tup[1]))
        print('======================================================')


    def get_sen_input_ids(self, test_sen):
        tokenized_sen = self.tokenizer.tokenize(test_sen)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokenized_sen)
        self.print_input_ids(test_sen, tokenized_sen, input_ids)
        return input_ids

    def get_para_input_ids(self, test_para):
        return [self.get_sen_input_ids(test_sen) for test_sen in test_para]

    def get_tokenizer_return(self, test_sen):
        encoded = self.tokenizer(test_sen, truncation=True, return_tensors='pt')
        print('========================================input ids============================================')
        print(encoded['input_ids'].numpy())
        print('=====================================attention_masks=========================================')
        print(encoded['attention_mask'].numpy())
        print('=============================================================================================')
        return encoded

    def process_text(text):
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
        
    def setEmbedding(text):
        processed_texts = process_text2(text)
        embeds = getBertEmbedding(processed_texts)
    