import torch
import torch.nn as nn 
class LSTMModelNoun(nn.Module):
    def __init__(self,   vocab_size_nouns, embedding_dim, hidden_dim, num_layers):
        super(LSTMModelNoun, self).__init__() 
        self.embedding_nouns = nn.Embedding(vocab_size_nouns, embedding_dim) 
        self.lstm_noun = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False ) 
        self.fc_nouns = nn.Linear(hidden_dim, vocab_size_nouns)

    def forward(self,x_noun): 
        x_noun = self.embedding_nouns(x_noun) 
        lstm_noun_out, _ = self.lstm_noun(x_noun) # lstm_noun değişkeni 
        logits_noun = self.fc_nouns(lstm_noun_out)
        return   logits_noun
    