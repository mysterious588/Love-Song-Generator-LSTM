import torch.nn as nn
import torch

class RNN(nn.Module):
    
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        super(RNN, self).__init__()
                
        # set class variables
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # define model layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first = True)
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, nn_input, hidden):
        """
        Forward propagation of the neural network
        :param nn_input: The input to the neural network
        :param hidden: The hidden state        
        :return: Two Tensors, the output of the neural network and the latest hidden state
        """
        nn_input = torch.tensor(nn_input).to(torch.int64)
        batch_size = nn_input.size(0)
        #print(nn_input.shape)
        embed_output = self.embedding(nn_input)
        lstm_output, hidden = self.lstm(embed_output, hidden)
        lstm_output = lstm_output.contiguous().view(-1, self.hidden_dim)
        nn_output = self.fc(lstm_output)
        nn_output = nn_output.view(batch_size, -1, self.output_size)
        output = nn_output[:, -1]
        # return one batch of output word scores and the hidden state
        return output, hidden
    
    
    def init_hidden(self, batch_size):

        from torch.autograd import Variable
        train_on_gpu = torch.cuda.is_available()
        
        if train_on_gpu:
            hidden = (Variable(next(self.parameters()).data.new(self.n_layers, batch_size, self.hidden_dim)).zero_().cuda(),
                      Variable(next(self.parameters()).data.new(self.n_layers, batch_size, self.hidden_dim)).zero_().cuda())
                
        else:
            hidden = (Variable(next(self.parameters()).data.new(self.n_layers, batch_size, self.hidden_dim)).zero_(),
                      Variable(next(self.parameters()).data.new(self.n_layers, batch_size, self.hidden_dim)).zero_())
                
        # initialize hidden state with zero weights, and move to GPU if available
        return hidden