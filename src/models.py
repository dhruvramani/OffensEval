import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class LSTMClassifier(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, embedding_length):
        super(LSTMClassifier, self).__init__()
        
        """
        Arguments
        ---------
        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        hidden_sie : Size of the hidden_state of the LSTM
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embeddding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 
        
        """
        
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        #self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        
        # NOTE : ignoring word_embedding as we already get pretrained embeddings
        #self.word_embeddings = nn.Embedding(vocab_size, embedding_length)# Initializing the look-up table.
        #self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False) # Assigning the look-up table to the pre-trained GloVe word embedding.
        self.lstm = nn.LSTM(self.embedding_length, self.hidden_size)
        self.label = nn.Linear(self.hidden_size, self.output_size)
        self.h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda()) # Initial hidden state of the LSTM
        self.c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda()) # Initial cell state of the LSTM
        
    def forward(self, input_sentence):
    
        """ 
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences, embedding_length)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)
        
        Returns
        -------
        Output of the linear layer containing logits for positive & negative class which receives its input as the final_hidden_state of the LSTM
        final_output.shape = (batch_size, output_size)
        
        """
        
        ''' Here we will map all the indexes present in the input sequence to the corresponding word vector using our pre-trained word_embedddins.'''
        #input = self.word_embeddings(input_sentence) # embedded input of shape = (batch_size, num_sequences,  embedding_length)
        input_sentence = input_sentence.permute(1, 0, 2) # input.size() = (num_sequences, batch_size, embedding_length)

        output, (final_hidden_state, final_cell_state) = self.lstm(input_sentence, (self.h_0, self.c_0))
        final_output = self.label(final_hidden_state[-1]) # final_hidden_state.size() = (1, batch_size, hidden_size) & final_output.size() = (batch_size, output_size)
        
        return final_output


class AttentionModel(torch.nn.Module):
    def __init__(self, batch_size, output_sizes, hidden_size, embedding_length):
        super(AttentionModel, self).__init__()
        
        """
        Arguments
        ---------
        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        hidden_sie : Size of the hidden_state of the LSTM
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embeddding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 
        
        --------
        
        """
        
        self.batch_size = batch_size
        self.output_sizes = output_sizes
        self.hidden_size = hidden_size
        self.embedding_length = embedding_length
        
        #self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
        #self.word_embeddings.weights = nn.Parameter(weights, requires_grad=False)
        self.lstm = nn.LSTM(embedding_length, hidden_size)
        self.label1 = nn.Linear(hidden_size, self.output_sizes[0])
        #self.label2 = nn.Linear(hidden_size, self.output_sizes[1])
        #self.label3 = nn.Linear(hidden_size, self.output_sizes[2])
        self.h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())
        self.c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())
        #self.attn_fc_layer = nn.Linear()
        
    def attention_net(self, lstm_output, final_state):

        """ 
        Now we will incorporate Attention mechanism in our LSTM model. In this new model, we will use attention to compute soft alignment score corresponding
        between each of the hidden_state and the last hidden_state of the LSTM. We will be using torch.bmm for the batch matrix multiplication.
        
        Arguments
        ---------
        
        lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.
        final_state : Final time-step hidden state (h_n) of the LSTM
        
        ---------
        
        Returns : It performs attention mechanism by first computing weights for each of the sequence present in lstm_output and and then finally computing the
                  new hidden state.
                  
        Tensor Size :
                    hidden.size() = (batch_size, hidden_size)
                    attn_weights.size() = (batch_size, num_seq)
                    soft_attn_weights.size() = (batch_size, num_seq)
                    new_hidden_state.size() = (batch_size, hidden_size)
                      
        """
        
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        
        return new_hidden_state
    
    def forward(self, input_sentence):
    
        """ 
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)
        
        Returns
        -------
        Output of the linear layer containing logits for pos & neg class which receives its input as the new_hidden_state which is basically the output of the Attention network.
        final_output.shape = (batch_size, output_size)
        
        """
        
        #input = self.word_embeddings(input_sentences)
        input_sentence = input_sentence.permute(1, 0, 2)
            
        output, (final_hidden_state, final_cell_state) = self.lstm(input_sentence, (self.h_0, self.c_0)) # final_hidden_state.size() = (1, batch_size, hidden_size) 
        output = output.permute(1, 0, 2) # output.size() = (batch_size, num_seq, hidden_size)
        
        attn_output = self.attention_net(output, final_hidden_state)
        logits1 = self.label1(attn_output)
        #logits2 = self.label2(attn_output)
        #logits3 = self.label3(attn_output)
        
        return logits1 #, logits2, logits3
