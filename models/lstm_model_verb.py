
import torch.nn as nn 
class LSTMModelVerb(nn.Module):
    def __init__(self, vocab_size_verbs,   embedding_dim, hidden_dim, num_layers):
        super(LSTMModelVerb, self).__init__()
        self.embedding_verbs = nn.Embedding(vocab_size_verbs, embedding_dim) 
        self.lstm_verb = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False )         
        self.fc_verbs = nn.Linear(hidden_dim, vocab_size_verbs) 

    def forward(self, x_verb ):
        x_verb = self.embedding_verbs(x_verb) 
        lstm_verb_out, _ = self.lstm_verb(x_verb) # lstm_verb değişkeni 
        logits_verb = self.fc_verbs(lstm_verb_out) 
        return logits_verb 