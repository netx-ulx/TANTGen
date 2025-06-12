import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    '''
    Implement the PE function.
    Positional encoding is used to give the model some information about the relative or absolute position of the tokens in the sequences.
    From "Attention Is All You Need" paper
    '''
    def __init__(self, d_model, max_len=5000): # TODO verify max_len
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]


class PacketTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=3, input_dim=4): # TODO: change input_dim
        """
        Initialize the PacketTransformer model.
        Args:
            d_model (int): Dimension of the model.
            nhead (int): Number of heads in the multi-head attention mechanism.
            num_layers (int): Number of encoder and decoder layers.
            input_dim (int): Dimension of the input features.
        """
        super(PacketTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead), num_layers) 

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead), num_layers)

        self.output_head = nn.Linear(d_model, input_dim)

    def forward(self, src, tgt, tgt_mask=None):
        # Encode
        src = self.embedding(src)
        src = self.pos_encoder(src)
        memory = self.encoder(src)

        # Decode
        tgt = self.embedding(tgt)
        tgt = self.pos_encoder(tgt)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)

        # Output head
        return self.output_head(output)
    

# TODO não sei se é necessário, mas é bom deixar aqui
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

def get_all_sentences(ds, lang):
    for pair in ds:
        yield pair['translation'][lang]

# Defining Tokenizer
def build_tokenizer(config, ds, lang):
    
    # Crating a file path for the tokenizer 
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    
    # Checking if Tokenizer already exists
    if not Path.exists(tokenizer_path): 
        
        # If it doesn't exist, we create a new one
        tokenizer = Tokenizer(WordLevel(unk_token = '[UNK]')) # Initializing a new world-level tokenizer
        tokenizer.pre_tokenizer = Whitespace() # We will split the text into tokens based on whitespace
        
        # Creating a trainer for the new tokenizer
        trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[PAD]", 
                                                     "[SOS]", "[EOS]"], min_frequency = 2) # Defining Word Level strategy and special tokens
        
        # Training new tokenizer on sentences from the dataset and language specified 
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer = trainer)
        tokenizer.save(str(tokenizer_path)) # Saving trained tokenizer to the file path specified at the beginning of the function
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path)) # If the tokenizer already exist, we load it
    return tokenizer # Returns the loaded tokenizer or the trained tokenizer