import string
import re
import torch
from torch import nn, utils
import numpy as np
from torch.autograd import Variable
import random
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from masked_cross_entropy import *
import torch.nn.functional as F
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
class Basic_Encoder(nn.Module):      
    def __init__(self, input_size, embedding_size, instruction_context_size, hidden_size, bidirectional):
        super(Basic_Encoder, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.instruction_context_size = instruction_context_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx = 0)
        self.lstm = nn.LSTM(embedding_size, hidden_size, bidirectional = self.bidirectional)
        self.out = nn.Sequential(nn.Linear(hidden_size * 2, instruction_context_size)) 
        

    def forward(self, input, input_lengths, hidden = None):
        embedded = self.embedding(input)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths) # pack
        outputs, hidden = self.lstm(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs) # unpack
        
        if self.bidirectional:
            hidden = self._cat_directions(hidden)
            
        hidden_0 = self.out(hidden[0].squeeze(0)) # only use the hidden state, not cell state
        return outputs, hidden_0
    
    def _cat_directions(self, hidden):
        def _cat(h):
            return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
            
        # LSTM hidden contains a tuple (hidden state, cell state)
        hidden = tuple([_cat(h) for h in hidden])
            
        return hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        hidden = weight.new(2, batch_size, self.hidden_size).zero_().to(device)
        cell = weight.new(2, batch_size, self.hidden_size).zero_().to(device)
        return (hidden, cell)

class Basic_Decoder(nn.Module):
    def __init__(self, embedding_size, instruction_context_size, hidden_size, output_size):
        super(Basic_Decoder, self).__init__()
        self.embedding_size = embedding_size
        self.instruction_context_size = instruction_context_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, embedding_size, padding_idx = 0)
        self.lstm = nn.LSTM(embedding_size + instruction_context_size, hidden_size)
        self.out = nn.Sequential(nn.Linear(hidden_size, output_size))

    def forward(self, input, context, hidden):
        batch_size = input.size(0)
        input = input.to(device)
        embedded = self.embedding(input).view(1, batch_size, self.embedding_size).to(device)
        decoder_input = torch.cat((embedded, context.unsqueeze(0)), dim = 2)
        output, hidden = self.lstm(decoder_input, hidden)
        output = self.out(output[0])  
        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        hidden = weight.new(1, batch_size, self.hidden_size).zero_().to(device)
        cell = weight.new(1, batch_size, self.hidden_size).zero_().to(device)
        return (hidden,cell)

class BasicModel(nn.Module):
    def __init__(self, input_size, embedding_size, instruction_context_size, encoder_hidden_size, decoder_hidden_size, output_size):
        super(BasicModel, self).__init__()
        self.input_size = input_size 
        self.embedding_size = embedding_size
        self.encoder_hidden_size = encoder_hidden_size
        self.instruction_context_size = instruction_context_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.EncoderRNN = Basic_Encoder(input_size, embedding_size, instruction_context_size, encoder_hidden_size, bidirectional = True)
        self.DecoderRNN = Basic_Decoder(embedding_size, instruction_context_size, decoder_hidden_size, output_size) 
    
    def _encode_seq(self, input, input_lengths, hidden):
        return self.EncoderRNN(input, input_lengths, hidden)
       
    def _decode(self, input, context, hidden):                
        return self.DecoderRNN(input, context, hidden)

    def train_batch(self, batch, optimizer, criterion, eval = False):
        """Updates model parameters, e.g. in a batch (suggested).

        TODO: implement this function. Suggested impleemntation is, for each
        item in the batch, compute the loss (cross entropy) over the entire sequence,
        then backpropagate the gradient of the average (over output sequence length)
        loss (as a dy.Expression). Remember to renew the graph at the beginning!

        In general, you will want to encode the inputs (utterances and world states), then
        decode a sequence of probability distributions over output tokens.
        """
        input_tensor = batch[0][:, :-28].to(device)
        target_tensor = batch[1].to(device)
        batch_size = input_tensor.size(0)
        # First: order the 0-idx padded batch by decreasing sequence length 
        input_lengths = torch.LongTensor([torch.max(input_tensor[i, :].data.nonzero()) + 1 for i in range(input_tensor.size()[0])])
        input_lengths, perm_idx = input_lengths.sort(0, descending=True)
        input_tensor = input_tensor[perm_idx][:, :input_lengths.max()]
        target_lengths = torch.LongTensor([torch.max(target_tensor[i,:].data.nonzero()) + 1 for i in range(target_tensor.size()[0])])
        target_tensor = target_tensor[perm_idx][:, :target_lengths.max()]
        
        max_input_length = input_lengths.max()
        max_target_length = target_lengths.max()
        input_lengths = input_lengths.tolist()
        target_lengths = target_lengths.tolist()
        
        input_tensor = Variable(input_tensor.transpose(0,1))
        target_tensor = Variable(target_tensor.transpose(0,1))
        
        optimizer.zero_grad()
        
        encoder_hidden = self.EncoderRNN.init_hidden(batch_size)
        encoder_outputs, encoder_hidden = self._encode_seq(input_tensor, input_lengths, encoder_hidden)
        decoder_input = torch.ones(batch_size, device = device).long()
        
        decoder_hidden = self.DecoderRNN.init_hidden(batch_size)      
        all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, self.output_size)).to(device)
        
        for di in range(max_target_length):
            decoder_output, decoder_hidden = self._decode(decoder_input, encoder_hidden, decoder_hidden)
            all_decoder_outputs[di] = decoder_output
            decoder_input = target_tensor[di]
       
        loss = criterion(all_decoder_outputs.transpose(0,1).transpose(1,2), target_tensor.transpose(0, 1).contiguous())
      
        if not eval:
            loss.backward() 
            optimizer.step()
            
        #for name, param in self.DecoderRNN.named_parameters():
        #    if param.requires_grad:
        #        print("\n")
        #        print (name, param.grad)
        
        return loss
                        

    def predict(self, input_tensor, instruction_level = True):
        """Returns a predicted sequence given an example.

        TODO: implement this function. You will want to encode the inputs, and
        then decode an action sequence. You are welcome to return the action sequence,
        or resulting world state from executing the sequence, or both, whichever works for your code.
        """
        input_tensor = input_tensor[:,:-28].to(device)
        input_lengths = torch.LongTensor([torch.max(input_tensor[i, :].data.nonzero()) + 1 for i in range(input_tensor.size()[0])])
        input_lengths, perm_idx = input_lengths.sort(0, descending=True)
        input_tensor = input_tensor[perm_idx][:, :input_lengths.max()]
        #print("input tensor: " + str(input_tensor))
        batch_size = 1
        
        max_input_length = input_lengths.max()
        input_lengths = input_lengths.tolist()
        
        input_tensor = Variable(input_tensor.transpose(0,1))
        
        encoder_hidden = self.EncoderRNN.init_hidden(batch_size)
        encoder_outputs, encoder_hidden = self._encode_seq(input_tensor, input_lengths, encoder_hidden)
        decoder_input = torch.ones(batch_size, device = device).long()

        decoder_hidden = self.DecoderRNN.init_hidden(batch_size)      
        
        decoded_word_indices = []
        for di in range(100):
            decoder_output, decoder_hidden = self._decode(decoder_input, encoder_hidden, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            #print(decoder_output.data)
            ni = topi[0][0]
            decoded_word_indices.append(ni.item())
            if ni == 2:
                break 
            decoder_input = Variable(torch.LongTensor([ni]))
    
        return decoded_word_indices
    
       
class Basic_Encoder2(nn.Module):      
    def __init__(self, input_size, embedding_size, instruction_context_size, hidden_size, bidirectional):
        super(Basic_Encoder2, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.instruction_context_size = instruction_context_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx = 0)
        self.lstm = nn.LSTM(embedding_size, hidden_size, bidirectional = self.bidirectional)
        self.out = nn.Sequential(nn.Linear(hidden_size * 2, instruction_context_size)) 
        self.out2 = nn.Sequential(nn.Linear(hidden_size * 2, instruction_context_size)) 
        

    def forward(self, input, input_lengths, hidden = None):
        embedded = self.embedding(input)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths) # pack
        outputs, hidden = self.lstm(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs) # unpack
        
        if self.bidirectional:
            hidden = self._cat_directions(hidden)
            
        hidden_0 = self.out(hidden[0].squeeze(0)).unsqueeze(0) # hidden
        hidden_1 = self.out(hidden[1].squeeze(0)).unsqueeze(0) # cell
    
        return outputs, (hidden_0,hidden_1)
    
    def _cat_directions(self, hidden):
        def _cat(h):
            return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
            
        # LSTM hidden contains a tuple (hidden state, cell state)
        hidden = tuple([_cat(h) for h in hidden])
            
        return hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        hidden = weight.new(2, batch_size, self.hidden_size).zero_().to(device)
        cell = weight.new(2, batch_size, self.hidden_size).zero_().to(device)
        return (hidden, cell)

class Basic_Decoder2(nn.Module):
    def __init__(self, embedding_size, instruction_context_size, output_size):
        super(Basic_Decoder2, self).__init__()
        self.embedding_size = embedding_size
        self.instruction_context_size = instruction_context_size
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, embedding_size, padding_idx = 0)
        self.lstm = nn.LSTM(embedding_size, instruction_context_size)
        self.out = nn.Sequential(nn.Linear(instruction_context_size, output_size))

    def forward(self, input, hidden):
        batch_size = input.size(0)
        input = input.to(device)
        embedded = self.embedding(input).view(1, batch_size, self.embedding_size).to(device)
        output, hidden = self.lstm(embedded, hidden)
        output = self.out(output[0])  
        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        hidden = weight.new(1, batch_size, self.hidden_size).zero_().to(device)
        cell = weight.new(1, batch_size, self.hidden_size).zero_().to(device)
        return (hidden,cell)

class BasicModel2(nn.Module):
    def __init__(self, input_size, embedding_size, instruction_context_size, encoder_hidden_size, output_size):
        super(BasicModel2, self).__init__()
        self.input_size = input_size 
        self.embedding_size = embedding_size
        self.encoder_hidden_size = encoder_hidden_size
        self.instruction_context_size = instruction_context_size
        self.output_size = output_size
        self.EncoderRNN = Basic_Encoder2(input_size, embedding_size, instruction_context_size, encoder_hidden_size, bidirectional = True)
        self.DecoderRNN = Basic_Decoder2(embedding_size, instruction_context_size, output_size) 
    
    def _encode_seq(self, input, input_lengths, hidden):
        return self.EncoderRNN(input, input_lengths, hidden)
       
    def _decode(self, input, hidden):                
        return self.DecoderRNN(input, hidden)

    def train_batch(self, batch, optimizer, criterion, eval = False):
        """Updates model parameters, e.g. in a batch (suggested).

        TODO: implement this function. Suggested impleemntation is, for each
        item in the batch, compute the loss (cross entropy) over the entire sequence,
        then backpropagate the gradient of the average (over output sequence length)
        loss (as a dy.Expression). Remember to renew the graph at the beginning!

        In general, you will want to encode the inputs (utterances and world states), then
        decode a sequence of probability distributions over output tokens.
        """
        input_tensor = batch[0][:, :-28].to(device)
        target_tensor = batch[1].to(device)
        batch_size = input_tensor.size(0)
        # First: order the 0-idx padded batch by decreasing sequence length 
        input_lengths = torch.LongTensor([torch.max(input_tensor[i, :].data.nonzero()) + 1 for i in range(input_tensor.size()[0])])
        input_lengths, perm_idx = input_lengths.sort(0, descending=True)
        input_tensor = input_tensor[perm_idx][:, :input_lengths.max()]
        target_lengths = torch.LongTensor([torch.max(target_tensor[i,:].data.nonzero()) + 1 for i in range(target_tensor.size()[0])])
        target_tensor = target_tensor[perm_idx][:, :target_lengths.max()]
        
        max_input_length = input_lengths.max()
        max_target_length = target_lengths.max()
        input_lengths = input_lengths.tolist()
        target_lengths = target_lengths.tolist()
        
        input_tensor = Variable(input_tensor.transpose(0,1))
        target_tensor = Variable(target_tensor.transpose(0,1))
        
        optimizer.zero_grad()
        
        encoder_hidden = self.EncoderRNN.init_hidden(batch_size)
        encoder_outputs, encoder_hidden = self._encode_seq(input_tensor, input_lengths, encoder_hidden)
        decoder_input = torch.ones(batch_size, device = device).long()
        
        decoder_hidden = encoder_hidden    
        all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, self.output_size)).to(device)
        
        for di in range(max_target_length):
            decoder_output, decoder_hidden = self._decode(decoder_input, decoder_hidden)
            all_decoder_outputs[di] = decoder_output
            decoder_input = target_tensor[di]
       
        loss = criterion(all_decoder_outputs.transpose(0,1).transpose(1,2), target_tensor.transpose(0, 1).contiguous())
      
        if not eval:
            loss.backward() 
            optimizer.step()
            
        #for name, param in self.DecoderRNN.named_parameters():
        #    if param.requires_grad:
        #        print("\n")
        #        print (name, param.grad)
        
        return loss
                        

    def predict(self, input_tensor, instruction_level = True):
        """Returns a predicted sequence given an example.

        TODO: implement this function. You will want to encode the inputs, and
        then decode an action sequence. You are welcome to return the action sequence,
        or resulting world state from executing the sequence, or both, whichever works for your code.
        """
        input_tensor = input_tensor[:,:-28].to(device)
        input_lengths = torch.LongTensor([torch.max(input_tensor[i, :].data.nonzero()) + 1 for i in range(input_tensor.size()[0])])
        input_lengths, perm_idx = input_lengths.sort(0, descending=True)
        input_tensor = input_tensor[perm_idx][:, :input_lengths.max()]
        #print("input tensor: " + str(input_tensor))
        batch_size = 1
        
        max_input_length = input_lengths.max()
        input_lengths = input_lengths.tolist()
        
        input_tensor = Variable(input_tensor.transpose(0,1))
        
        encoder_hidden = self.EncoderRNN.init_hidden(batch_size)
        encoder_outputs, encoder_hidden = self._encode_seq(input_tensor, input_lengths, encoder_hidden)
        decoder_input = torch.ones(batch_size, device = device).long()
        
        decoder_hidden = encoder_hidden    
        
        decoded_word_indices = []
        for di in range(100):
            decoder_output, decoder_hidden = self._decode(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            #print(decoder_output.data)
            ni = topi[0][0]
            decoded_word_indices.append(ni.item())
            if ni == 2:
                break 
            decoder_input = Variable(torch.LongTensor([ni]))
    
        return decoded_word_indices
    
class Environment_Encoder(nn.Module):      
    def __init__(self, input_size, embedding_size, environment_context_size, hidden_size):
        super(Environment_Encoder, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.environment_context_size = environment_context_size
        self.hidden_size = hidden_size
        self.hidden_size = environment_context_size
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx = 0)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first = True)
        #self.out = nn.Sequential(nn.Linear(hidden_size, environment_context_size)) 
      
    def forward_one_hot(self, input):
        one_hot = torch.nn.functional.one_hot(input, 7) # size=(7,4,n)
        return one_hot.flatten() # should be a 7 x 4 x 7 = 196 vector

    def forward(self, input, hidden = None):    
        embedded = self.embedding(input)
        outputs, hidden = self.lstm(embedded, hidden)
        #hidden_0 = self.out(hidden[0].squeeze(0)) # only use the hidden state, not cell state
        #return outputs, hidden_0
        return outputs, hidden[0].squeeze(0)

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        hidden = weight.new(1, batch_size, self.hidden_size).zero_().to(device)
        cell =  weight.new(1, batch_size, self.hidden_size).zero_().to(device)
        return (hidden, cell)
        
class Part2_Encoder(nn.Module):      
    def __init__(self, input_size, embedding_size, instruction_context_size, hidden_size, bidirectional):
        super(Part2_Encoder, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.instruction_context_size = instruction_context_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx = 0)
        self.lstm = nn.LSTM(embedding_size, hidden_size, bidirectional = self.bidirectional)
        self.out = nn.Sequential(nn.Linear(hidden_size * 2, instruction_context_size)) # out layer is only invoked if LSTM is bidirectional
        
    def forward(self, input, input_lengths, hidden = None):
        embedded = self.embedding(input)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths) # pack
        outputs, hidden = self.lstm(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs) # unpack
        
        if self.bidirectional:
            hidden = self._cat_directions(hidden)
            hidden_0 = self.out(hidden[0].squeeze(0)) # only use the hidden state, not cell state
            return outputs, hidden_0
        else:
            return outputs, hidden[0].squeeze(0) # only use the hidden state, not cell state
    
    def _cat_directions(self, hidden):
        def _cat(h):
            return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
            
        # LSTM hidden contains a tuple (hidden state, cell state)
        hidden = tuple([_cat(h) for h in hidden])
            
        return hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        hidden = weight.new(2, batch_size, self.hidden_size).zero_().to(device)
        cell = weight.new(2, batch_size, self.hidden_size).zero_().to(device)
        return (hidden, cell)

class Part2_Decoder(nn.Module):
    def __init__(self, embedding_size, instruction_context_size, environment_context_size, hidden_size, output_size):
        super(Part2_Decoder, self).__init__()
        self.embedding_size = embedding_size
        self.instruction_context_size = instruction_context_size
        self.environment_context_size = environment_context_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, embedding_size, padding_idx = 0)
        #self.dropout = nn.Dropout(p=0.2)
        self.lstm = nn.LSTM(embedding_size + instruction_context_size + (environment_context_size * 7), hidden_size)
        self.out = nn.Sequential(nn.Linear(hidden_size, output_size))

        
    def forward(self, input, instruction_context, environment_context, hidden):
        batch_size = input.size(0)   
        input = input.to(device)                       
        input = input.unsqueeze(0)  
        embedded = self.embedding(input).to(device) 
        #embedded = self.dropout(embedded)
        decoder_input = torch.cat((embedded, instruction_context.unsqueeze(0), environment_context.unsqueeze(0)), dim = 2)
        output, hidden = self.lstm(decoder_input, hidden)
        output = self.out(output) 
        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        hidden = weight.new(1, batch_size, self.hidden_size).zero_().to(device)
        cell = weight.new(1, batch_size, self.hidden_size).zero_().to(device)
        return (hidden,cell)
    
class Part2Model(nn.Module):
    def __init__(self, input_size, word_embedding_size, action_embedding_size, instruction_context_size, encoder_hidden_size, environment_context_size, environment_hidden_size, decoder_hidden_size, output_size, one_hot = False):
        super(Part2Model, self).__init__()
        self.input_size = input_size 
        self.word_embedding_size = word_embedding_size
        self.action_embedding_size = action_embedding_size
        self.encoder_hidden_size = encoder_hidden_size
        self.instruction_context_size = instruction_context_size
        self.environment_hidden_size = environment_hidden_size
        self.environment_context_size = environment_context_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.one_hot = one_hot
        self.EncoderRNN = Part2_Encoder(input_size, word_embedding_size, instruction_context_size, encoder_hidden_size, bidirectional = False)
        self.EnvironmentEncoderRNN = Environment_Encoder(input_size = 7, embedding_size = 7, environment_context_size = environment_context_size, hidden_size = environment_hidden_size)
        if self.one_hot:
            self.DecoderRNN = Part2_Decoder(action_embedding_size, instruction_context_size, 28, decoder_hidden_size, output_size)
        else:
            self.DecoderRNN = Part2_Decoder(action_embedding_size, instruction_context_size, environment_context_size, decoder_hidden_size, output_size) 
    
    def _encode_environment(self, input):
        return self.EnvironmentEncoderRNN(input, hidden = None)
    
    def _encode_seq(self, input, input_lengths):
        return self.EncoderRNN(input, input_lengths, hidden = None)
       
    def _decode(self, input, instruction_context, environment_context, hidden):                
        return self.DecoderRNN(input, instruction_context, environment_context, hidden)

    def train_batch(self, batch, optimizer, criterion, eval = False):
        """Updates model parameters, e.g. in a batch (suggested).

        TODO: implement this function. Suggested impleemntation is, for each
        item in the batch, compute the loss (cross entropy) over the entire sequence,
        then backpropagate the gradient of the average (over output sequence length)
        loss (as a dy.Expression). Remember to renew the graph at the beginning!

        In general, you will want to encode the inputs (utterances and world states), then
        decode a sequence of probability distributions over output tokens.
        """
        input_tensor = batch[0][:, :-28].to(device)
        environment_tensor = batch[0][:, -28:].to(device)
        target_tensor = batch[1].to(device)
        batch_size = input_tensor.size(0)
        # First: order the 0-idx padded batch by decreasing sequence length 
        input_lengths = torch.LongTensor([torch.max(input_tensor[i, :].data.nonzero()) + 1 for i in range(input_tensor.size()[0])])
        input_lengths, perm_idx = input_lengths.sort(0, descending=True)
        input_tensor = input_tensor[perm_idx][:, :input_lengths.max()]
        
        environment_tensor = environment_tensor[perm_idx]
        
        target_lengths = torch.LongTensor([torch.max(target_tensor[i,:].data.nonzero()) + 1 for i in range(target_tensor.size()[0])])
        target_tensor = target_tensor[perm_idx][:, :target_lengths.max()]
        
        max_input_length = input_lengths.max()
        max_target_length = target_lengths.max()
        input_lengths = input_lengths.tolist()
        target_lengths = target_lengths.tolist()
        
        input_tensor = Variable(input_tensor.transpose(0,1))
        target_tensor = Variable(target_tensor.transpose(0,1))
        
        optimizer.zero_grad()
        
        encoder_outputs, encoder_hidden = self._encode_seq(input_tensor, input_lengths)
        
        environment_tensor = list(torch.split(environment_tensor, split_size_or_sections = 4, dim = 1))
        environment_tensor = torch.stack(environment_tensor, dim = 0).transpose(0,1)
        environment_context = []
        
        for i in range(environment_tensor.size(0)): 
            input = environment_tensor[i]
            if self.one_hot:
                environment_context.append(self.EnvironmentEncoderRNN.forward_one_hot(input))
            else:
                outputs, hidden = self._encode_environment(input)
                environment_context.append(hidden.flatten())
                
     
        environment_context= torch.stack(environment_context, dim = 0) 
        instruction_context = encoder_hidden
        
        decoder_input = torch.ones(batch_size, device = device).long()  
        decoder_hidden = None  
        
        all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, self.output_size)).to(device)
        
        for di in range(max_target_length):
            decoder_output, decoder_hidden = self._decode(decoder_input, instruction_context, environment_context, decoder_hidden)
            decoder_output = decoder_output.squeeze(0)
            all_decoder_outputs[di] = decoder_output
            decoder_input = target_tensor[di]
        
        loss = criterion(all_decoder_outputs.transpose(0,1).transpose(1,2), target_tensor.transpose(0, 1).contiguous())
      
        if not eval:
            loss.backward() 
            optimizer.step()
            
        #for name, param in self.DecoderRNN.named_parameters():
        #    if param.requires_grad:
        #        print("\n")
        #        print (name, param.grad)
        
        return loss
                        

    def predict(self, input_tensor, instruction_level = True):
        """Returns a predicted sequence given an example.

        TODO: implement this function. You will want to encode the inputs, and
        then decode an action sequence. You are welcome to return the action sequence,
        or resulting world state from executing the sequence, or both, whichever works for your code.
        """
        environment_tensor = input_tensor[:, -28:].to(device)
        input_tensor = input_tensor[:,:-28].to(device)
        input_lengths = torch.LongTensor([torch.max(input_tensor[i, :].data.nonzero()) + 1 for i in range(input_tensor.size()[0])])
        input_lengths, perm_idx = input_lengths.sort(0, descending=True)
        input_tensor = input_tensor[perm_idx][:, :input_lengths.max()]
        
        environment_tensor = environment_tensor[perm_idx]
         
        batch_size = 1
        
        max_input_length = input_lengths.max()
        input_lengths = input_lengths.tolist()
        
        input_tensor = Variable(input_tensor.transpose(0,1))
        
        encoder_outputs, encoder_hidden = self._encode_seq(input_tensor, input_lengths)
        
        environment_tensor = list(torch.split(environment_tensor, split_size_or_sections = 4, dim = 1))
        environment_tensor = torch.stack(environment_tensor, dim = 0).transpose(0,1)
        environment_context = []
        
        for i in range(environment_tensor.size(0)): 
            input = environment_tensor[i]
            if self.one_hot:
                environment_context.append(self.EnvironmentEncoderRNN.forward_one_hot(input))
            else:
                outputs, hidden = self._encode_environment(input)
                environment_context.append(hidden.flatten())
     
        environment_context= torch.stack(environment_context, dim = 0)
        instruction_context = encoder_hidden
        
        decoder_input = torch.ones(batch_size, device = device).long()
        decoder_hidden = None    
        
        decoded_word_indices = []
        for di in range(100):
            decoder_output, decoder_hidden = self._decode(decoder_input, instruction_context, environment_context, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            #print(decoder_output.data)
            ni = topi[0][0]
            decoded_word_indices.append(ni.item())
            if ni == 2:
                break 
            decoder_input = Variable(torch.LongTensor([ni]))
    
        return decoded_word_indices
  
     
class Part2_Encoder(nn.Module):      
    def __init__(self, input_size, embedding_size, instruction_context_size, hidden_size, bidirectional):
        super(Part2_Encoder, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.instruction_context_size = instruction_context_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx = 0)
        self.lstm = nn.LSTM(embedding_size, hidden_size, bidirectional = self.bidirectional)
        self.out = nn.Sequential(nn.Linear(hidden_size * 2, instruction_context_size)) # out layer is only invoked if LSTM is bidirectional
        
    def forward(self, input, input_lengths, hidden = None):
        embedded = self.embedding(input)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths) # pack
        outputs, hidden = self.lstm(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs) # unpack
        
        if self.bidirectional:
            hidden = self._cat_directions(hidden)
            hidden_0 = self.out(hidden[0].squeeze(0)) # only use the hidden state, not cell state
            return outputs, hidden_0
        else:
            return outputs, hidden[0].squeeze(0) # only use the hidden state, not cell state
    
    def _cat_directions(self, hidden):
        def _cat(h):
            return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
            
        # LSTM hidden contains a tuple (hidden state, cell state)
        hidden = tuple([_cat(h) for h in hidden])
            
        return hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        hidden = weight.new(2, batch_size, self.hidden_size).zero_().to(device)
        cell = weight.new(2, batch_size, self.hidden_size).zero_().to(device)
        return (hidden, cell)

class Part2_Decoder(nn.Module):
    def __init__(self, embedding_size, instruction_context_size, environment_context_size, hidden_size, output_size):
        super(Part2_Decoder, self).__init__()
        self.embedding_size = embedding_size
        self.instruction_context_size = instruction_context_size
        self.environment_context_size = environment_context_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, embedding_size, padding_idx = 0)
        #self.dropout = nn.Dropout(p=0.2)
        self.lstm = nn.LSTM(embedding_size + instruction_context_size + (environment_context_size * 7), hidden_size)
        self.out = nn.Sequential(nn.Linear(hidden_size, output_size))

        
    def forward(self, input, instruction_context, environment_context, hidden):
        batch_size = input.size(0)   
        input = input.to(device)                       
        input = input.unsqueeze(0)  
        embedded = self.embedding(input).to(device) 
        #embedded = self.dropout(embedded)
        decoder_input = torch.cat((embedded, instruction_context.unsqueeze(0), environment_context.unsqueeze(0)), dim = 2)
        output, hidden = self.lstm(decoder_input, hidden)
        output = self.out(output) 
        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        hidden = weight.new(1, batch_size, self.hidden_size).zero_().to(device)
        cell = weight.new(1, batch_size, self.hidden_size).zero_().to(device)
        return (hidden,cell)
    
class Part2Model(nn.Module):
    def __init__(self, input_size, word_embedding_size, action_embedding_size, instruction_context_size, encoder_hidden_size, environment_context_size, environment_hidden_size, decoder_hidden_size, output_size, one_hot = False):
        super(Part2Model, self).__init__()
        self.input_size = input_size 
        self.word_embedding_size = word_embedding_size
        self.action_embedding_size = action_embedding_size
        self.encoder_hidden_size = encoder_hidden_size
        self.instruction_context_size = instruction_context_size
        self.environment_hidden_size = environment_hidden_size
        self.environment_context_size = environment_context_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.one_hot = one_hot
        self.EncoderRNN = Part2_Encoder(input_size, word_embedding_size, instruction_context_size, encoder_hidden_size, bidirectional = False)
        self.EnvironmentEncoderRNN = Environment_Encoder(input_size = 7, embedding_size = 7, environment_context_size = environment_context_size, hidden_size = environment_hidden_size)
        if self.one_hot:
            self.DecoderRNN = Part2_Decoder(action_embedding_size, instruction_context_size, 28, decoder_hidden_size, output_size)
        else:
            self.DecoderRNN = Part2_Decoder(action_embedding_size, instruction_context_size, environment_context_size, decoder_hidden_size, output_size) 
    
    def _encode_environment(self, input):
        return self.EnvironmentEncoderRNN(input, hidden = None)
    
    def _encode_seq(self, input, input_lengths):
        return self.EncoderRNN(input, input_lengths, hidden = None)
       
    def _decode(self, input, instruction_context, environment_context, hidden):                
        return self.DecoderRNN(input, instruction_context, environment_context, hidden)

    def train_batch(self, batch, optimizer, criterion, eval = False):
        """Updates model parameters, e.g. in a batch (suggested).

        TODO: implement this function. Suggested impleemntation is, for each
        item in the batch, compute the loss (cross entropy) over the entire sequence,
        then backpropagate the gradient of the average (over output sequence length)
        loss (as a dy.Expression). Remember to renew the graph at the beginning!

        In general, you will want to encode the inputs (utterances and world states), then
        decode a sequence of probability distributions over output tokens.
        """
        input_tensor = batch[0][:, :-28].to(device)
        environment_tensor = batch[0][:, -28:].to(device)
        target_tensor = batch[1].to(device)
        batch_size = input_tensor.size(0)
        # First: order the 0-idx padded batch by decreasing sequence length 
        input_lengths = torch.LongTensor([torch.max(input_tensor[i, :].data.nonzero()) + 1 for i in range(input_tensor.size()[0])])
        input_lengths, perm_idx = input_lengths.sort(0, descending=True)
        input_tensor = input_tensor[perm_idx][:, :input_lengths.max()]
        
        environment_tensor = environment_tensor[perm_idx]
        
        target_lengths = torch.LongTensor([torch.max(target_tensor[i,:].data.nonzero()) + 1 for i in range(target_tensor.size()[0])])
        target_tensor = target_tensor[perm_idx][:, :target_lengths.max()]
        
        max_input_length = input_lengths.max()
        max_target_length = target_lengths.max()
        input_lengths = input_lengths.tolist()
        target_lengths = target_lengths.tolist()
        
        input_tensor = Variable(input_tensor.transpose(0,1))
        target_tensor = Variable(target_tensor.transpose(0,1))
        
        optimizer.zero_grad()
        
        encoder_outputs, encoder_hidden = self._encode_seq(input_tensor, input_lengths)
        
        environment_tensor = list(torch.split(environment_tensor, split_size_or_sections = 4, dim = 1))
        environment_tensor = torch.stack(environment_tensor, dim = 0).transpose(0,1)
        environment_context = []
        
        for i in range(environment_tensor.size(0)): 
            input = environment_tensor[i]
            if self.one_hot:
                environment_context.append(self.EnvironmentEncoderRNN.forward_one_hot(input))
            else:
                outputs, hidden = self._encode_environment(input)
                environment_context.append(hidden.flatten())
                
     
        environment_context= torch.stack(environment_context, dim = 0) 
        instruction_context = encoder_hidden
        
        decoder_input = torch.ones(batch_size, device = device).long()  
        decoder_hidden = None  
        
        all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, self.output_size)).to(device)
        
        for di in range(max_target_length):
            decoder_output, decoder_hidden = self._decode(decoder_input, instruction_context, environment_context, decoder_hidden)
            decoder_output = decoder_output.squeeze(0)
            all_decoder_outputs[di] = decoder_output
            decoder_input = target_tensor[di]
        
        loss = criterion(all_decoder_outputs.transpose(0,1).transpose(1,2), target_tensor.transpose(0, 1).contiguous())
      
        if not eval:
            loss.backward() 
            optimizer.step()
            
        #for name, param in self.DecoderRNN.named_parameters():
        #    if param.requires_grad:
        #        print("\n")
        #        print (name, param.grad)
        
        return loss
                        

    def predict(self, input_tensor, instruction_level = True):
        """Returns a predicted sequence given an example.

        TODO: implement this function. You will want to encode the inputs, and
        then decode an action sequence. You are welcome to return the action sequence,
        or resulting world state from executing the sequence, or both, whichever works for your code.
        """
        environment_tensor = input_tensor[:, -28:].to(device)
        input_tensor = input_tensor[:,:-28].to(device)
        input_lengths = torch.LongTensor([torch.max(input_tensor[i, :].data.nonzero()) + 1 for i in range(input_tensor.size()[0])])
        input_lengths, perm_idx = input_lengths.sort(0, descending=True)
        input_tensor = input_tensor[perm_idx][:, :input_lengths.max()]
        
        environment_tensor = environment_tensor[perm_idx]
         
        batch_size = 1
        
        max_input_length = input_lengths.max()
        input_lengths = input_lengths.tolist()
        
        input_tensor = Variable(input_tensor.transpose(0,1))
        
        encoder_outputs, encoder_hidden = self._encode_seq(input_tensor, input_lengths)
        
        environment_tensor = list(torch.split(environment_tensor, split_size_or_sections = 4, dim = 1))
        environment_tensor = torch.stack(environment_tensor, dim = 0).transpose(0,1)
        environment_context = []
        
        for i in range(environment_tensor.size(0)): 
            input = environment_tensor[i]
            if self.one_hot:
                environment_context.append(self.EnvironmentEncoderRNN.forward_one_hot(input))
            else:
                outputs, hidden = self._encode_environment(input)
                environment_context.append(hidden.flatten())
     
        environment_context= torch.stack(environment_context, dim = 0)
        instruction_context = encoder_hidden
        
        decoder_input = torch.ones(batch_size, device = device).long()
        decoder_hidden = None    
        
        decoded_word_indices = []
        for di in range(100):
            decoder_output, decoder_hidden = self._decode(decoder_input, instruction_context, environment_context, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            #print(decoder_output.data)
            ni = topi[0][0]
            decoded_word_indices.append(ni.item())
            if ni == 2:
                break 
            decoder_input = Variable(torch.LongTensor([ni]))
    
        return decoded_word_indices
    
    
    


class Part3_BahdanauAttention_Encoder(nn.Module):      
    def __init__(self, input_size, embedding_size, hidden_size, bidirectional):
        super(Part3_BahdanauAttention_Encoder, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        #self.instruction_context_size = instruction_context_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx = 0)
        self.lstm = nn.LSTM(embedding_size, hidden_size, bidirectional = self.bidirectional)
        #self.out = nn.Sequential(nn.Linear(hidden_size * 2, instruction_context_size)) # out layer is only invoked if LSTM is bidirectional
        
    def forward(self, input, input_lengths, hidden = None):
        embedded = self.embedding(input)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths) # pack
        outputs, hidden = self.lstm(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs) # unpack
        
        #if self.bidirectional:
        #    hidden = self._cat_directions(hidden)
        #    hidden_0 = self.out(hidden[0].squeeze(0)) # only use the hidden state, not cell state
        #    return outputs, hidden_0
        #else:
        return outputs, hidden
    
    def _cat_directions(self, hidden):
        def _cat(h):
            return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
            
        # LSTM hidden contains a tuple (hidden state, cell state)
        hidden = tuple([_cat(h) for h in hidden])
            
        return hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        hidden = weight.new(2, batch_size, self.hidden_size).zero_().to(device)
        cell = weight.new(2, batch_size, self.hidden_size).zero_().to(device)
        return (hidden, cell)

class Part3_BahdanauAttention_Decoder(nn.Module):
    def __init__(self, embedding_size, environment_context_size, hidden_size, output_size):
        super(Part3_BahdanauAttention_Decoder, self).__init__()
        self.embedding_size = embedding_size
        self.environment_context_size = environment_context_size
        self.hidden_size = hidden_size  # hidden size = instruction context size/encoder output size
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, embedding_size, padding_idx = 0)
        #self.dropout = nn.Dropout(p=0.2)
        self.fc_hidden = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_encoder = nn.Linear(self.hidden_size, self.hidden_size)  
        self.attn_combine = nn.Linear(self.hidden_size, self.hidden_size)
        self.alignment = nn.Linear(self.hidden_size, 1)
        
        self.lstm = nn.LSTM(embedding_size + hidden_size + (environment_context_size * 7), hidden_size)
        self.out = nn.Sequential(nn.Linear(hidden_size, output_size))

        
    def forward(self, input, encoder_outputs, environment_context, hidden):
        
        batch_size = input.size(0)   
        input = input.to(device)                       
        input = input.unsqueeze(0)  
        embedded = self.embedding(input).to(device) 
        #embedded = self.dropout(embedded)
        # attention mechanism
        
        #weight = nn.Parameter(torch.FloatTensor(batch_size, self.hidden_size)).unsqueeze(2).to(device) 

        #weight = nn.Parameter(torch.FloatTensor(batch_size, self.hidden_size)).to(device)
        
        x = torch.tanh(self.fc_hidden(hidden[0]) + self.fc_encoder(encoder_outputs)).transpose(0,1)
        
        #alignment_scores = x.bmm(weight.unsqueeze(2)).squeeze(2)
        alignment_scores = self.alignment(x).squeeze(2)
    

        encoder_outputs = encoder_outputs.transpose(0,1)
        attn_weights = F.softmax(alignment_scores, dim = 1)
        attn_weights = attn_weights.unsqueeze(1)
        
        instruction_context = torch.bmm(attn_weights, encoder_outputs).transpose(0,1)
 
        

        decoder_input = torch.cat((embedded, instruction_context, environment_context.unsqueeze(0)), dim = 2)
        output, hidden = self.lstm(decoder_input, hidden)
        output = self.out(output) 
        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        hidden = weight.new(1, batch_size, self.hidden_size).zero_().to(device)
        cell = weight.new(1, batch_size, self.hidden_size).zero_().to(device)
        return (hidden,cell)
    
class Part3_BahdanauAttention_Model(nn.Module):
    def __init__(self, input_size, word_embedding_size, action_embedding_size, encoder_hidden_size, environment_context_size, environment_hidden_size, decoder_hidden_size, output_size, one_hot = False):
        super(Part3_BahdanauAttention_Model, self).__init__()
        self.input_size = input_size 
        self.word_embedding_size = word_embedding_size
        self.action_embedding_size = action_embedding_size
        self.encoder_hidden_size = encoder_hidden_size
        #self.instruction_context_size = instruction_context_size
        self.environment_hidden_size = environment_hidden_size
        self.environment_context_size = environment_context_size
        # decoder_hidden_size should be the same as the encoder's hidden size!!
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.one_hot = one_hot
        self.EncoderRNN = Part3_BahdanauAttention_Encoder(input_size, word_embedding_size, encoder_hidden_size, bidirectional = False)
        self.EnvironmentEncoderRNN = Environment_Encoder(input_size = 7, embedding_size = 7, environment_context_size = environment_context_size, hidden_size = environment_hidden_size)
        if self.one_hot:
            self.DecoderRNN = Part3_BahdanauAttention_Decoder(action_embedding_size, 28, decoder_hidden_size, output_size)
        else:
            self.DecoderRNN = Part3_BahdanauAttention_Decoder(action_embedding_size, environment_context_size, decoder_hidden_size, output_size) 
    
    def _encode_environment(self, input):
        return self.EnvironmentEncoderRNN(input, hidden = None)
    
    def _encode_seq(self, input, input_lengths):
        return self.EncoderRNN(input, input_lengths, hidden = None)
       
    def _decode(self, input, encoder_outputs, environment_context, hidden):                
        return self.DecoderRNN(input, encoder_outputs, environment_context, hidden)

    def train_batch(self, batch, optimizer, criterion, eval = False):
        """Updates model parameters, e.g. in a batch (suggested).

        TODO: implement this function. Suggested impleemntation is, for each
        item in the batch, compute the loss (cross entropy) over the entire sequence,
        then backpropagate the gradient of the average (over output sequence length)
        loss (as a dy.Expression). Remember to renew the graph at the beginning!

        In general, you will want to encode the inputs (utterances and world states), then
        decode a sequence of probability distributions over output tokens.
        """
        input_tensor = batch[0][:, :-28].to(device)
        environment_tensor = batch[0][:, -28:].to(device)
        target_tensor = batch[1].to(device)
        batch_size = input_tensor.size(0)
        # First: order the 0-idx padded batch by decreasing sequence length 
        input_lengths = torch.LongTensor([torch.max(input_tensor[i, :].data.nonzero()) + 1 for i in range(input_tensor.size()[0])])
        input_lengths, perm_idx = input_lengths.sort(0, descending=True)
        input_tensor = input_tensor[perm_idx][:, :input_lengths.max()]
        
        environment_tensor = environment_tensor[perm_idx]
        
        target_lengths = torch.LongTensor([torch.max(target_tensor[i,:].data.nonzero()) + 1 for i in range(target_tensor.size()[0])])
        target_tensor = target_tensor[perm_idx][:, :target_lengths.max()]
        
        max_input_length = input_lengths.max()
        max_target_length = target_lengths.max()
        input_lengths = input_lengths.tolist()
        target_lengths = target_lengths.tolist()
        
        input_tensor = Variable(input_tensor.transpose(0,1))
        target_tensor = Variable(target_tensor.transpose(0,1))
        
        optimizer.zero_grad()
        
        encoder_outputs, encoder_hidden = self._encode_seq(input_tensor, input_lengths)
        
        environment_tensor = list(torch.split(environment_tensor, split_size_or_sections = 4, dim = 1))
        environment_tensor = torch.stack(environment_tensor, dim = 0).transpose(0,1)
        environment_context = []
        
        for i in range(environment_tensor.size(0)): 
            input = environment_tensor[i]
            if self.one_hot:
                environment_context.append(self.EnvironmentEncoderRNN.forward_one_hot(input))
            else:
                outputs, hidden = self._encode_environment(input)
                environment_context.append(hidden.flatten())
                
     
        environment_context= torch.stack(environment_context, dim = 0) 
        
        decoder_input = torch.ones(batch_size, device = device).long()  
        decoder_hidden = self.DecoderRNN.init_hidden(batch_size)  
        all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, self.output_size)).to(device)
        
        for di in range(max_target_length):
            decoder_output, decoder_hidden = self._decode(decoder_input, encoder_outputs, environment_context, decoder_hidden)
            decoder_output = decoder_output.squeeze(0)
            all_decoder_outputs[di] = decoder_output
            decoder_input = target_tensor[di]
        
        loss = criterion(all_decoder_outputs.transpose(0,1).transpose(1,2), target_tensor.transpose(0, 1).contiguous())
        if torch.isnan(loss):
            print("nan loss! aborting training \n")
            exit()
            
        if not eval:
            loss.backward() 
            optimizer.step()
            
        #for name, param in self.DecoderRNN.named_parameters():
        #    if param.requires_grad:
        #        print("\n")
        #        print (name, param.grad)
        
        return loss
                        

    def predict(self, input_tensor, instruction_level = True):
        """Returns a predicted sequence given an example.

        TODO: implement this function. You will want to encode the inputs, and
        then decode an action sequence. You are welcome to return the action sequence,
        or resulting world state from executing the sequence, or both, whichever works for your code.
        """
        environment_tensor = input_tensor[:, -28:].to(device)
        input_tensor = input_tensor[:,:-28].to(device)
        input_lengths = torch.LongTensor([torch.max(input_tensor[i, :].data.nonzero()) + 1 for i in range(input_tensor.size()[0])])
        input_lengths, perm_idx = input_lengths.sort(0, descending=True)
        input_tensor = input_tensor[perm_idx][:, :input_lengths.max()]
        
        environment_tensor = environment_tensor[perm_idx]
         
        batch_size = 1
        
        max_input_length = input_lengths.max()
        input_lengths = input_lengths.tolist()
        
        input_tensor = Variable(input_tensor.transpose(0,1))
        
        encoder_outputs, encoder_hidden = self._encode_seq(input_tensor, input_lengths)
        
        environment_tensor = list(torch.split(environment_tensor, split_size_or_sections = 4, dim = 1))
        environment_tensor = torch.stack(environment_tensor, dim = 0).transpose(0,1)
        environment_context = []
        
        for i in range(environment_tensor.size(0)): 
            input = environment_tensor[i]
            if self.one_hot:
                environment_context.append(self.EnvironmentEncoderRNN.forward_one_hot(input))
            else:
                outputs, hidden = self._encode_environment(input)
                environment_context.append(hidden.flatten())
     
        environment_context= torch.stack(environment_context, dim = 0)
        
        decoder_input = torch.ones(batch_size, device = device).long()
        decoder_hidden = self.DecoderRNN.init_hidden(batch_size)  
        
        decoded_word_indices = []
        for di in range(100):
            decoder_output, decoder_hidden = self._decode(decoder_input, encoder_outputs, environment_context, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            #print(decoder_output.data)
            ni = topi[0][0]
            decoded_word_indices.append(ni.item())
            if ni == 2:
                break 
            decoder_input = Variable(torch.LongTensor([ni]))
    
        return decoded_word_indices
    
    
    
    
class Part3_BasicDotProductAttention_Encoder(nn.Module):      
    def __init__(self, input_size, embedding_size, hidden_size, bidirectional):
        super(Part3_BasicDotProductAttention_Encoder, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        #self.dropout = nn.Dropout(p=0.2)
        #self.instruction_context_size = instruction_context_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx = 0)
        self.lstm = nn.LSTM(embedding_size, hidden_size, bidirectional = self.bidirectional)
        #self.out = nn.Sequential(nn.Linear(hidden_size * 2, instruction_context_size)) # out layer is only invoked if LSTM is bidirectional
        
    def forward(self, input, input_lengths, hidden = None):
        embedded = self.embedding(input)
        #embedded = self.dropout(embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths) # pack
        outputs, hidden = self.lstm(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs) # unpack
        
        #if self.bidirectional:
        #    hidden = self._cat_directions(hidden)
        #    hidden_0 = self.out(hidden[0].squeeze(0)) # only use the hidden state, not cell state
        #    return outputs, hidden_0
        #else:
        return outputs, hidden
    
    def _cat_directions(self, hidden):
        def _cat(h):
            return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
            
        # LSTM hidden contains a tuple (hidden state, cell state)
        hidden = tuple([_cat(h) for h in hidden])
            
        return hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        hidden = weight.new(2, batch_size, self.hidden_size).zero_().to(device)
        cell = weight.new(2, batch_size, self.hidden_size).zero_().to(device)
        return (hidden, cell)

class Part3_BasicDotProductAttention_Decoder(nn.Module):
    def __init__(self, embedding_size, environment_context_size, hidden_size, output_size):
        super(Part3_BasicDotProductAttention_Decoder, self).__init__()
        self.embedding_size = embedding_size
        self.environment_context_size = environment_context_size
        self.hidden_size = hidden_size  # hidden size = instruction context size/encoder output size
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, embedding_size, padding_idx = 0)
        #self.dropout = nn.Dropout(p=0.2)
        
        
        self.lstm = nn.LSTM(embedding_size + hidden_size + (environment_context_size * 7), hidden_size)
        self.out = nn.Sequential(nn.Linear(hidden_size, output_size))

        
    def forward(self, input, encoder_outputs, environment_context, hidden):
        
        batch_size = input.size(0)   
        input = input.to(device)                       
        input = input.unsqueeze(0)  
        embedded = self.embedding(input).to(device) 
        #embedded = self.dropout(embedded)
        # attention mechanism
        encoder_outputs = encoder_outputs.transpose(0,1)
 
        attn_scores = torch.bmm(encoder_outputs,hidden[0].transpose(0,1).transpose(1,2))
        attn_weights = F.softmax(attn_scores, dim = 1)
        encoder_outputs = encoder_outputs.transpose(1,2)
        instruction_context = torch.bmm(encoder_outputs, attn_weights).squeeze(2)

        

        decoder_input = torch.cat((embedded, instruction_context.unsqueeze(0), environment_context.unsqueeze(0)), dim = 2)
        output, hidden = self.lstm(decoder_input, hidden)
        output = self.out(output) 
        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        hidden = weight.new(1, batch_size, self.hidden_size).zero_().to(device)
        cell = weight.new(1, batch_size, self.hidden_size).zero_().to(device)
        return (hidden,cell)
    
class Part3_BasicDotProductAttention_Model(nn.Module):
    def __init__(self, input_size, word_embedding_size, action_embedding_size, encoder_hidden_size, environment_context_size, environment_hidden_size, decoder_hidden_size, output_size, one_hot = False):
        super(Part3_BasicDotProductAttention_Model, self).__init__()
        self.input_size = input_size 
        self.word_embedding_size = word_embedding_size
        self.action_embedding_size = action_embedding_size
        self.encoder_hidden_size = encoder_hidden_size
        #self.instruction_context_size = instruction_context_size
        self.environment_hidden_size = environment_hidden_size
        self.environment_context_size = environment_context_size
        # decoder_hidden_size should be the same as the encoder's hidden size!!
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.one_hot = one_hot
        self.EncoderRNN = Part3_BasicDotProductAttention_Encoder(input_size, word_embedding_size, encoder_hidden_size, bidirectional = False)
        self.EnvironmentEncoderRNN = Environment_Encoder(input_size = 7, embedding_size = 7, environment_context_size = environment_context_size, hidden_size = environment_hidden_size)
        if self.one_hot:
            self.DecoderRNN = Part3_BasicDotProductAttention_Decoder(action_embedding_size, 28, decoder_hidden_size, output_size)
        else:
            self.DecoderRNN = Part3_BasicDotProductAttention_Decoder(action_embedding_size, environment_context_size, decoder_hidden_size, output_size) 
    
    def _encode_environment(self, input):
        return self.EnvironmentEncoderRNN(input, hidden = None)
    
    def _encode_seq(self, input, input_lengths):
        return self.EncoderRNN(input, input_lengths, hidden = None)
       
    def _decode(self, input, encoder_outputs, environment_context, hidden):                
        return self.DecoderRNN(input, encoder_outputs, environment_context, hidden)

    def train_batch(self, batch, optimizer, criterion, eval = False):
        """Updates model parameters, e.g. in a batch (suggested).

        TODO: implement this function. Suggested impleemntation is, for each
        item in the batch, compute the loss (cross entropy) over the entire sequence,
        then backpropagate the gradient of the average (over output sequence length)
        loss (as a dy.Expression). Remember to renew the graph at the beginning!

        In general, you will want to encode the inputs (utterances and world states), then
        decode a sequence of probability distributions over output tokens.
        """
        input_tensor = batch[0][:, :-28].to(device)
        environment_tensor = batch[0][:, -28:].to(device)
        target_tensor = batch[1].to(device)
        batch_size = input_tensor.size(0)
        # First: order the 0-idx padded batch by decreasing sequence length 
        input_lengths = torch.LongTensor([torch.max(input_tensor[i, :].data.nonzero()) + 1 for i in range(input_tensor.size()[0])])
        input_lengths, perm_idx = input_lengths.sort(0, descending=True)
        input_tensor = input_tensor[perm_idx][:, :input_lengths.max()]
        
        environment_tensor = environment_tensor[perm_idx]
        
        target_lengths = torch.LongTensor([torch.max(target_tensor[i,:].data.nonzero()) + 1 for i in range(target_tensor.size()[0])])
        target_tensor = target_tensor[perm_idx][:, :target_lengths.max()]
        
        max_input_length = input_lengths.max()
        max_target_length = target_lengths.max()
        input_lengths = input_lengths.tolist()
        target_lengths = target_lengths.tolist()
        
        input_tensor = Variable(input_tensor.transpose(0,1))
        target_tensor = Variable(target_tensor.transpose(0,1))
        
        optimizer.zero_grad()
        
        encoder_outputs, encoder_hidden = self._encode_seq(input_tensor, input_lengths)
        
        environment_tensor = list(torch.split(environment_tensor, split_size_or_sections = 4, dim = 1))
        environment_tensor = torch.stack(environment_tensor, dim = 0).transpose(0,1)
        environment_context = []
        
        for i in range(environment_tensor.size(0)): 
            input = environment_tensor[i]
            if self.one_hot:
                environment_context.append(self.EnvironmentEncoderRNN.forward_one_hot(input))
            else:
                outputs, hidden = self._encode_environment(input)
                environment_context.append(hidden.flatten())
                
     
        environment_context= torch.stack(environment_context, dim = 0) 
        
        decoder_input = torch.ones(batch_size, device = device).long()  
        decoder_hidden = self.DecoderRNN.init_hidden(batch_size)  
        all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, self.output_size)).to(device)
        
        for di in range(max_target_length):
            decoder_output, decoder_hidden = self._decode(decoder_input, encoder_outputs, environment_context, decoder_hidden)
            decoder_output = decoder_output.squeeze(0)
            all_decoder_outputs[di] = decoder_output
            decoder_input = target_tensor[di]
        
        loss = criterion(all_decoder_outputs.transpose(0,1).transpose(1,2), target_tensor.transpose(0, 1).contiguous())
        if torch.isnan(loss):
            print("nan loss! aborting training \n")
            exit()
            
        if not eval:
            loss.backward() 
            optimizer.step()
            
        #for name, param in self.DecoderRNN.named_parameters():
        #    if param.requires_grad:
        #        print("\n")
        #        print (name, param.grad)
        
        return loss
                        

    def predict(self, input_tensor, instruction_level = True):
        """Returns a predicted sequence given an example.

        TODO: implement this function. You will want to encode the inputs, and
        then decode an action sequence. You are welcome to return the action sequence,
        or resulting world state from executing the sequence, or both, whichever works for your code.
        """
        environment_tensor = input_tensor[:, -28:].to(device)
        input_tensor = input_tensor[:,:-28].to(device)
        input_lengths = torch.LongTensor([torch.max(input_tensor[i, :].data.nonzero()) + 1 for i in range(input_tensor.size()[0])])
        input_lengths, perm_idx = input_lengths.sort(0, descending=True)
        input_tensor = input_tensor[perm_idx][:, :input_lengths.max()]
        
        environment_tensor = environment_tensor[perm_idx]
         
        batch_size = 1
        
        max_input_length = input_lengths.max()
        input_lengths = input_lengths.tolist()
        
        input_tensor = Variable(input_tensor.transpose(0,1))
        
        encoder_outputs, encoder_hidden = self._encode_seq(input_tensor, input_lengths)
        
        environment_tensor = list(torch.split(environment_tensor, split_size_or_sections = 4, dim = 1))
        environment_tensor = torch.stack(environment_tensor, dim = 0).transpose(0,1)
        environment_context = []
        
        for i in range(environment_tensor.size(0)): 
            input = environment_tensor[i]
            if self.one_hot:
                environment_context.append(self.EnvironmentEncoderRNN.forward_one_hot(input))
            else:
                outputs, hidden = self._encode_environment(input)
                environment_context.append(hidden.flatten())
     
        environment_context= torch.stack(environment_context, dim = 0)
        
        decoder_input = torch.ones(batch_size, device = device).long()
        decoder_hidden = self.DecoderRNN.init_hidden(batch_size)  
        
        decoded_word_indices = []
        for di in range(100):
            decoder_output, decoder_hidden = self._decode(decoder_input, encoder_outputs, environment_context, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            #print(decoder_output.data)
            ni = topi[0][0]
            decoded_word_indices.append(ni.item())
            if ni == 2:
                break 
            decoder_input = Variable(torch.LongTensor([ni]))
    
        return decoded_word_indices
    
    
    
    
    
    
class Part4_BahdanauAttention_Encoder(nn.Module):      
    def __init__(self, input_size, embedding_size, hidden_size, bidirectional):
        super(Part4_BahdanauAttention_Encoder, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        #self.instruction_context_size = instruction_context_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx = 0)
        self.lstm = nn.LSTM(embedding_size, hidden_size, bidirectional = self.bidirectional)
        #self.out = nn.Sequential(nn.Linear(hidden_size * 2, instruction_context_size)) # out layer is only invoked if LSTM is bidirectional
        
    def forward(self, input, input_lengths, hidden = None):
        embedded = self.embedding(input)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, enforce_sorted = False) # pack
        outputs, hidden = self.lstm(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs) # unpack
        
        #if self.bidirectional:
        #    hidden = self._cat_directions(hidden)
        #    hidden_0 = self.out(hidden[0].squeeze(0)) # only use the hidden state, not cell state
        #    return outputs, hidden_0
        #else:
        return outputs, hidden
    
    def _cat_directions(self, hidden):
        def _cat(h):
            return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
            
        # LSTM hidden contains a tuple (hidden state, cell state)
        hidden = tuple([_cat(h) for h in hidden])
            
        return hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        hidden = weight.new(2, batch_size, self.hidden_size).zero_().to(device)
        cell = weight.new(2, batch_size, self.hidden_size).zero_().to(device)
        return (hidden, cell)

class Part4_BahdanauAttention_Decoder(nn.Module):
    def __init__(self, embedding_size, environment_context_size, hidden_size, output_size):
        super(Part4_BahdanauAttention_Decoder, self).__init__()
        self.embedding_size = embedding_size
        self.environment_context_size = environment_context_size
        self.hidden_size = hidden_size  # hidden size = instruction context size/encoder output size
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, embedding_size, padding_idx = 0)
        #self.dropout = nn.Dropout(p=0.2)
        self.fc_hidden_cur = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_encoder_cur = nn.Linear(self.hidden_size, self.hidden_size)  
        self.attn_combine_cur = nn.Linear(self.hidden_size, self.hidden_size)
        self.alignment_cur = nn.Linear(self.hidden_size, 1)
        
        self.fc_hidden_prev = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_encoder_prev = nn.Linear(self.hidden_size, self.hidden_size)  
        self.attn_combine_prev = nn.Linear(self.hidden_size, self.hidden_size)
        self.alignment_prev = nn.Linear(self.hidden_size, 1)
        
        self.lstm = nn.LSTM(embedding_size + hidden_size + hidden_size + (environment_context_size * 7), hidden_size)
        self.out = nn.Sequential(nn.Linear(hidden_size, output_size))

        
    def forward(self, input, previous_outputs, encoder_outputs, environment_context, hidden):
        
        batch_size = input.size(0)   
        input = input.to(device)                       
        input = input.unsqueeze(0)  
        embedded = self.embedding(input).to(device) 
    
        # attention for previous_outputs
        x_prev = torch.tanh(self.fc_hidden_prev(hidden[0]) + self.fc_encoder_prev(previous_outputs)).transpose(0,1)
        alignment_scores_prev = self.alignment_prev(x_prev).squeeze(2)
    

        previous_outputs = previous_outputs.transpose(0,1)
        attn_weights_prev = F.softmax(alignment_scores_prev, dim = 1)
        attn_weights_prev = attn_weights_prev.unsqueeze(1)
        
        instruction_context_prev = torch.bmm(attn_weights_prev, previous_outputs).transpose(0,1)
        
        # attention for encoder_outputs
        x_cur = torch.tanh(self.fc_hidden_cur(hidden[0]) + self.fc_encoder_cur(encoder_outputs)).transpose(0,1)
        alignment_scores_cur = self.alignment_cur(x_cur).squeeze(2)

        encoder_outputs = encoder_outputs.transpose(0,1)
        attn_weights_cur = F.softmax(alignment_scores_cur, dim = 1)
        attn_weights_cur = attn_weights_cur.unsqueeze(1)
        
        instruction_context_cur = torch.bmm(attn_weights_cur, encoder_outputs).transpose(0,1)
        

        decoder_input = torch.cat((embedded, instruction_context_prev, instruction_context_cur, environment_context.unsqueeze(0)), dim = 2)
        output, hidden = self.lstm(decoder_input, hidden)
        output = self.out(output) 
        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        hidden = weight.new(1, batch_size, self.hidden_size).zero_().to(device)
        cell = weight.new(1, batch_size, self.hidden_size).zero_().to(device)
        return (hidden,cell)
    
class Part4_BahdanauAttention_Model(nn.Module):
    def __init__(self, input_size, word_embedding_size, action_embedding_size, encoder_hidden_size, environment_context_size, environment_hidden_size, decoder_hidden_size, output_size, one_hot = False):
        super(Part4_BahdanauAttention_Model, self).__init__()
        self.input_size = input_size 
        self.word_embedding_size = word_embedding_size
        self.action_embedding_size = action_embedding_size
        self.encoder_hidden_size = encoder_hidden_size
        #self.instruction_context_size = instruction_context_size
        self.environment_hidden_size = environment_hidden_size
        self.environment_context_size = environment_context_size
        # decoder_hidden_size should be the same as the encoder's hidden size!!
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.one_hot = one_hot
        self.EncoderRNN_current = Part4_BahdanauAttention_Encoder(input_size, word_embedding_size, encoder_hidden_size, bidirectional = False)
        self.EncoderRNN_previous = Part4_BahdanauAttention_Encoder(input_size, word_embedding_size, encoder_hidden_size, bidirectional = False)
        self.EnvironmentEncoderRNN = Environment_Encoder(input_size = 7, embedding_size = 7, environment_context_size = environment_context_size, hidden_size = environment_hidden_size)
        if self.one_hot:
            self.DecoderRNN = Part4_BahdanauAttention_Decoder(action_embedding_size, 28, decoder_hidden_size, output_size)
        else:
            self.DecoderRNN = Part4_BahdanauAttention_Decoder(action_embedding_size, environment_context_size, decoder_hidden_size, output_size) 
    
    def _encode_environment(self, input):
        return self.EnvironmentEncoderRNN(input, hidden = None)
    
    def _encode_seq_current(self, input, input_lengths):
        return self.EncoderRNN_current(input, input_lengths, hidden = None)
    
    def _encode_seq_previous(self, input, input_lengths):
        return self.EncoderRNN_previous(input, input_lengths, hidden = None)
       
    def _decode(self, input, previous_outputs, encoder_outputs, environment_context, hidden):                
        return self.DecoderRNN(input, previous_outputs, encoder_outputs, environment_context, hidden)

    def train_batch(self, batch, optimizer, criterion, eval = False):
        """Updates model parameters, e.g. in a batch (suggested).

        TODO: implement this function. Suggested impleemntation is, for each
        item in the batch, compute the loss (cross entropy) over the entire sequence,
        then backpropagate the gradient of the average (over output sequence length)
        loss (as a dy.Expression). Remember to renew the graph at the beginning!

        In general, you will want to encode the inputs (utterances and world states), then
        decode a sequence of probability distributions over output tokens.
        """

        divider = int((batch[0].size(1) - 28) / 2)
        prev_tensor = batch[0][:, :divider].to(device)
        input_tensor = batch[0][:, divider:-28].to(device)
        environment_tensor = batch[0][:, -28:].to(device)
        target_tensor = batch[1].to(device)
        batch_size = batch[0].size(0)
 
        input_lengths = torch.LongTensor([torch.max(input_tensor[i, :].data.nonzero()) + 1 for i in range(input_tensor.size()[0])])
        input_tensor = input_tensor[:, :input_lengths.max()]
        
        prev_lengths = torch.LongTensor([torch.max(prev_tensor[i, :].data.nonzero()) + 1 for i in range(prev_tensor.size()[0])])
        prev_tensor = prev_tensor[:, :prev_lengths.max()]
        
        
        target_lengths = torch.LongTensor([torch.max(target_tensor[i,:].data.nonzero()) + 1 for i in range(target_tensor.size()[0])])
        target_tensor = target_tensor[:, :target_lengths.max()]
        
        max_input_length = input_lengths.max()
        max_target_length = target_lengths.max()
        input_lengths = input_lengths.tolist()
        target_lengths = target_lengths.tolist()
        
        prev_tensor = Variable(prev_tensor.transpose(0,1))
        input_tensor = Variable(input_tensor.transpose(0,1))
        target_tensor = Variable(target_tensor.transpose(0,1))
        
        optimizer.zero_grad()
        
        encoder_outputs, encoder_hidden = self._encode_seq_current(input_tensor, input_lengths)
  
        prev_outputs, prev_hidden = self._encode_seq_previous(prev_tensor, prev_lengths)
        
        environment_tensor = list(torch.split(environment_tensor, split_size_or_sections = 4, dim = 1))
        environment_tensor = torch.stack(environment_tensor, dim = 0).transpose(0,1)
        environment_context = []
        
        for i in range(environment_tensor.size(0)): 
            input = environment_tensor[i]
            if self.one_hot:
                environment_context.append(self.EnvironmentEncoderRNN.forward_one_hot(input))
            else:
                outputs, hidden = self._encode_environment(input)
                environment_context.append(hidden.flatten())
                
     
        environment_context= torch.stack(environment_context, dim = 0) 
        
        decoder_input = torch.ones(batch_size, device = device).long()  
        decoder_hidden = self.DecoderRNN.init_hidden(batch_size)  
        all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, self.output_size)).to(device)
        
        for di in range(max_target_length):
            decoder_output, decoder_hidden = self._decode(decoder_input, prev_outputs, encoder_outputs, environment_context, decoder_hidden)
            decoder_output = decoder_output.squeeze(0)
            all_decoder_outputs[di] = decoder_output
            decoder_input = target_tensor[di]
        
        loss = criterion(all_decoder_outputs.transpose(0,1).transpose(1,2), target_tensor.transpose(0, 1).contiguous())
        if torch.isnan(loss):
            print("nan loss! aborting training \n")
            exit()
            
        if not eval:
            loss.backward() 
            optimizer.step()
            
        #for name, param in self.DecoderRNN.named_parameters():
        #    if param.requires_grad:
        #        print("\n")
        #        print (name, param.grad)
        
        return loss
                        

    def predict(self, input_tensor, instruction_level = True):
        """Returns a predicted sequence given an example.

        TODO: implement this function. You will want to encode the inputs, and
        then decode an action sequence. You are welcome to return the action sequence,
        or resulting world state from executing the sequence, or both, whichever works for your code.
        """
        divider = int((input_tensor.size(1) - 28) / 2)
        prev_tensor = input_tensor[:, :divider].to(device)
        environment_tensor = input_tensor[:, -28:].to(device)
        input_tensor = input_tensor[:, divider:-28].to(device)
        batch_size = 1
 
        input_lengths = torch.LongTensor([torch.max(input_tensor[i, :].data.nonzero()) + 1 for i in range(input_tensor.size()[0])])
        input_tensor = input_tensor[:, :input_lengths.max()]
        
        prev_lengths = torch.LongTensor([torch.max(prev_tensor[i, :].data.nonzero()) + 1 for i in range(prev_tensor.size()[0])])
        prev_tensor = prev_tensor[:, :prev_lengths.max()]
        
        
        max_input_length = input_lengths.max()
        input_lengths = input_lengths.tolist()
   
        
        prev_tensor = Variable(prev_tensor.transpose(0,1))
        input_tensor = Variable(input_tensor.transpose(0,1))

        
        encoder_outputs, encoder_hidden = self._encode_seq_current(input_tensor, input_lengths)
  
        prev_outputs, prev_hidden = self._encode_seq_previous(prev_tensor, prev_lengths)
        
        environment_tensor = list(torch.split(environment_tensor, split_size_or_sections = 4, dim = 1))
        environment_tensor = torch.stack(environment_tensor, dim = 0).transpose(0,1)
        environment_context = []
        
        for i in range(environment_tensor.size(0)): 
            input = environment_tensor[i]
            if self.one_hot:
                environment_context.append(self.EnvironmentEncoderRNN.forward_one_hot(input))
            else:
                outputs, hidden = self._encode_environment(input)
                environment_context.append(hidden.flatten())
                
     
        environment_context= torch.stack(environment_context, dim = 0) 
        
        decoder_input = torch.ones(batch_size, device = device).long()  
        decoder_hidden = self.DecoderRNN.init_hidden(batch_size)   
        
        decoded_word_indices = []
        for di in range(100):
            decoder_output, decoder_hidden = self._decode(decoder_input, prev_outputs, encoder_outputs, environment_context, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            #print(decoder_output.data)
            ni = topi[0][0]
            decoded_word_indices.append(ni.item())
            if ni == 2:
                break 
            decoder_input = Variable(torch.LongTensor([ni]))
    
        return decoded_word_indices