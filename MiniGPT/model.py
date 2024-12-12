## Building and training a bigram language model
from functools import partial
import math

import torch
import torch.nn as nn
from einops import einsum, reduce, rearrange
import torch.nn.functional as F


class BigramLanguageModel(nn.Module):
    """
    Class definition for a simple bigram language model.
    """

    def __init__(self, config):
        """
        Initialize the bigram language model.

        The model should have the following layers:
        1. An embedding layer that maps tokens to embeddings. (self.embeddings)
        2. A linear layer that maps embeddings to logits. (self.linear) **set bias to True**
        3. A dropout layer. (self.dropout)

        NOTE : PLEASE KEEP OF EACH LAYER AS PROVIDED BELOW TO FACILITATE TESTING.
        """

        super().__init__()
        vocab_size = config.vocab_size
        embedding_dim = config.embed_dim
        dropout_prob = config.dropout
        # ========= TODO : START ========= #

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size, bias=True)
        self.dropout = nn.Dropout(dropout_prob)
 
        # ========= TODO : END ========= #

        self.apply(self._init_weights)

    def forward(self, x):
        """
        Forward pass of the bigram language model.

        Args:
        x : torch.Tensor
            A tensor of shape (batch_size, 1) containing the input tokens.

        Output:
        torch.Tensor
            A tensor of shape (batch_size, vocab_size) containing the logits.
        """

        # ========= TODO : START ========= #

        # Get embeddings for input ids
        embeddings = self.embeddings(x)

        # Apply dropout
        embeddings = self.dropout(embeddings)

        # Get logits from linear layer
        logits = self.linear(embeddings)

        return logits

        # ========= TODO : END ========= #

    def _init_weights(self, module):
        """
        Weight initialization for better convergence.

        NOTE : You do not need to modify this function.
        """

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def generate(self, context, max_new_tokens=100):
        """
        Use the model to generate new tokens given a context.
        We will perform multinomial sampling which is very similar to greedy sampling
        but instead of taking the token with the highest probability, we sample the next token from a multinomial distribution.


        Args:
        context : List[int]
            A list of integers (tokens) representing the context.
        max_new_tokens : int
            The maximum number of new tokens to generate.

        Output:
        List[int]
            A list of integers (tokens) representing the generated tokens.
        """

        ### ========= TODO : START ========= ###

        # context = torch.tensor(context, dtype=torch.long).unsqueeze(0)  # Convert context to tensor and add batch dimension
        # generated_tokens = context.tolist()[0]  # Initialize generated tokens with the context
        # new_tokens = []

        # for _ in range(max_new_tokens):
        #     logits = self.forward(context)  # Get logits from the model
        #     next_token_logits = logits[:, -1, :]  # Get logits for the last token in the sequence

        #     probabilities = torch.softmax(next_token_logits, dim=-1)  # Convert logits to probabilities
        #     next_token = torch.multinomial(probabilities, num_samples=1).item()  # Sample next token from multinomial distribution
        #     new_tokens.append(next_token)
        #     generated_tokens.append(next_token)  # Append the generated token to the sequence
        #     context = torch.tensor(generated_tokens, dtype=torch.long).unsqueeze(0)  # Update context with the new token

        # return next_token
        
        
        
        # context = context.to(self.device)  # Ensure context is on the correct device
        # generated_tokens = context.tolist()  # Initialize generated tokens with the context
    
        # for _ in range(max_new_tokens):
        #     logits = self.forward(context)  # Get logits from the model
    
        #     # Ensure logits are 3-dimensional: [batch_size, sequence_length, vocab_size]
        #     if logits.dim() == 2:  # If logits are 2-dimensional, add the batch and sequence dimensions
        #         logits = logits.unsqueeze(0)
        #     elif logits.dim() == 1:  # If logits are 1-dimensional, add batch and sequence dimensions
        #         logits = logits.unsqueeze(0).unsqueeze(0)
    
        #     next_token_logits = logits[:, -1, :]  # Get logits for the last token in the sequence
    
        #     probabilities = torch.softmax(next_token_logits, dim=-1)  # Convert logits to probabilities
    
        #     # Ensure probabilities are 2-dimensional [batch_size, vocab_size]
        #     if probabilities.dim() == 3:
        #         probabilities = probabilities.squeeze(0)
            
        #     next_token = torch.multinomial(probabilities, num_samples=1).item()  # Sample next token from multinomial distribution
    
        #     generated_tokens.append(next_token)  # Append the generated token to the sequence
    
        #     context = torch.tensor(generated_tokens, dtype=torch.long).unsqueeze(0)  # Update context with the new token on the correct device
    
        # return generated_tokens  # Return all generated tokens
        context = context.clone().detach()
        generated_tokens = context.tolist()
        
        if len(context.shape) == 1:
            context = context.unsqueeze(0).to(next(self.parameters()).device)
        
        context_length = 10
        
        for _ in range(max_new_tokens):
            current_context = context[:, -context_length:]  # Maintain the window size
    
            logits = self(current_context)
    
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, next_token), dim=1)
    
        return context.squeeze()

        
        
        # context = torch.tensor(context, dtype=torch.long).unsqueeze(0).to(self.pos.device)  # Convert context to tensor, add batch dimension, and move to device
        # generated_tokens = context.tolist()[0]  # Initialize generated tokens with the context
    
        # for _ in range(max_new_tokens):
        #     logits = self.forward(context)  # Get logits from the model
        #     next_token_logits = logits[:, -1, :]  # Get logits for the last token in the sequence
    
        #     probabilities = torch.softmax(next_token_logits, dim=-1)  # Convert logits to probabilities
        #     next_token = torch.multinomial(probabilities, num_samples=1).item()  # Sample next token from multinomial distribution
        #     generated_tokens.append(next_token)  # Append the generated token to the sequence
        #     context = torch.tensor(generated_tokens, dtype=torch.long).unsqueeze(0).to(self.pos.device)  # Update context with the new token and move to device
        
    
        # return generated_tokens
        # context = torch.tensor(context, dtype=torch.long).unsqueeze(0).to(self.pos.device)  # Convert context to tensor, add batch dimension, and move to device
        # generated_tokens = context.tolist()[0]  # Initialize generated tokens with the context
    
        # for _ in range(max_new_tokens):
        #     logits = self.forward(context)  # Get logits from the model
        #     next_token_logits = logits[:, -1, :]  # Get logits for the last token in the sequence
    
        #     probabilities = torch.softmax(next_token_logits, dim=-1)  # Convert logits to probabilities
        #     next_token = torch.multinomial(probabilities, num_samples=1).item()  # Sample next token from multinomial distribution
        #     generated_tokens.append(next_token)  # Append the generated token to the sequence
        #     context = torch.tensor(generated_tokens, dtype=torch.long).unsqueeze(0).to(self.pos.device)  # Update context with the new token and move to device
    
        # return generated_tokens

        ### ========= TODO : END ========= ###


class SingleHeadAttention(nn.Module):
    """
    Class definition for Single Head Causal Self Attention Layer.

    As in Attention is All You Need (https://arxiv.org/pdf/1706.03762)

    """

    def __init__(
        self,
        input_dim,
        output_key_query_dim=None,
        output_value_dim=None,
        dropout=0.1,
        max_len=512,
    ):
        """
        Initialize the Single Head Attention Layer.

        The model should have the following layers:
        1. A linear layer for key. (self.key) **set bias to False**
        2. A linear layer for query. (self.query) **set bias to False**
        3. A linear layer for value. (self.value) # **set bias to False**
        4. A dropout layer. (self.dropout)
        5. A causal mask. (self.causal_mask) This should be registered as a buffer.
        NOTE : Please make sure that the causal mask is upper triangular and not lower triangular (this helps in setting up the test cases, )

         NOTE : PLEASE KEEP OF EACH LAYER AS PROVIDED BELOW TO FACILITATE TESTING.
        """
        super().__init__()

        self.input_dim = input_dim
        if output_key_query_dim:
            self.output_key_query_dim = output_key_query_dim
        else:
            self.output_key_query_dim = input_dim

        if output_value_dim:
            self.output_value_dim = output_value_dim
        else:
            self.output_value_dim = input_dim

        causal_mask = None  # You have to implement this, currently just a placeholder

        # ========= TODO : START ========= #

        self.key = nn.Linear(self.input_dim, self.output_key_query_dim, bias=False)
        self.query = nn.Linear(self.input_dim,self.output_key_query_dim, bias=False)
        self.value = nn.Linear(self.input_dim, self.output_value_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        causal_mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        causal_mask = causal_mask.to(dtype=torch.float32)  # ensure correct dtype if needed

        # ========= TODO : END ========= #

        self.register_buffer(
            "causal_mask", causal_mask
        )  # Registering as buffer to avoid backpropagation

    def forward(self, x):
        """
        Forward pass of the Single Head Attention Layer.

        Args:
        x : torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the input tokens.

        Output:
        torch.Tensor
            A tensor of shape (batch_size, num_tokens, output_value_dim) containing the output tokens.
        """

        # ========= TODO : START ========= #
        
        batch_size, num_tokens, token_dim = x.shape
        
        # print("x.shape:", batch_size, num_tokens, token_dim)

        key = self.key(x) # Shape: (batch_size, num_tokens, output_key_query_dim)
        query = self.query(x) # Shape: (batch_size, num_tokens, output_key_query_dim)
        value = self.value(x) # Shape: (batch_size, num_tokens, output_value_dim)
        
        

        if num_tokens > self.causal_mask.shape[0]:
            raise ValueError(f"Input sequence length ({num_tokens}) exceeds the maximum length for which the mask was created ({self.causal_mask.shape[0]}).")
    
        # Adjust mask to the current sequence length
        current_mask = self.causal_mask[:num_tokens, :num_tokens]
        
        # Compute scores
        scores = torch.matmul(query, key.transpose(-2, -1) / math.sqrt(self.output_key_query_dim))
        scores = scores.masked_fill(current_mask == 1, float('-inf'))
        
        # Apply softmax to get the attention weights
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to the values
        output = torch.matmul(attn_weights, value)
        
        return output

        # ========= TODO : END ========= #


class MultiHeadAttention(nn.Module):
    """
    Class definition for Multi Head Attention Layer.

    As in Attention is All You Need (https://arxiv.org/pdf/1706.03762)
    """

    def __init__(self, input_dim, num_heads, dropout=0.1) -> None:
        """
        Initialize the Multi Head Attention Layer.

        The model should have the following layers:
        1. Multiple SingleHeadAttention layers. (self.head_{i}) Use setattr to dynamically set the layers.
        2. A linear layer for output. (self.out) **set bias to True**
        3. A dropout layer. (self.dropout) Apply dropout to the output of the out layer.

        NOTE : PLEASE KEEP OF EACH LAYER AS PROVIDED BELOW TO FACILITATE TESTING.
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads

        # ========= TODO : START ========= #
        # per_head_dim = self.input_dim // self.numheads
        self.per_head_dim = self.input_dim // self.num_heads

        # Dynamically adding SingleHeadAttention heads
        # for i in range(num_heads):
        #     head = SingleHeadAttention(input_dim=per_head_dim, output_key_query_dim=per_head_dim)
        #     setattr(self, f'head_{i}', head)
        # self.out = nn.Linear(self.input_dim, self.input_dim, bias=True)
        # self.dropout = nn.Dropout(dropout)
        self.per_head_dim = self.input_dim // self.num_heads

        # Dynamically adding SingleHeadAttention heads
        for i in range(num_heads):
            head = SingleHeadAttention(
                input_dim=self.input_dim,
                output_key_query_dim=self.per_head_dim,
                output_value_dim=self.per_head_dim,
                dropout=dropout
            )
            setattr(self, f'head_{i}', head)
        self.out = nn.Linear(self.input_dim, self.input_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

        # ========= TODO : END ========= #

    def forward(self, x):
        """
        Forward pass of the Multi Head Attention Layer.

        Args:
        x : torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the input tokens.

        Output:
        torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the output tokens.
        """

        # ========= TODO : START ========= #

        # head_outputs = [getattr(self, f'head_{i}')(x) for i in range(self.num_heads)]

        # # Concatenating all head outputs
        # concat_outputs = torch.cat(head_outputs, dim=-1)

        # # Passing concatenated outputs through the output linear layer and dropout
        # output = self.out(concat_outputs)
        # output = self.dropout(output)
        batch_size, num_tokens, token_dim = x.shape

        # Split input for each head and concatenate the outputs
        head_outputs = []
        for i in range(self.num_heads):
            head = getattr(self, f'head_{i}')
            head_output = head(x)
            head_outputs.append(head_output)
        
        # Concatenate all head outputs
        concat_outputs = torch.cat(head_outputs, dim=-1)

        # Passing concatenated outputs through the output linear layer and dropout
        output = self.out(concat_outputs)
        output = self.dropout(output)

        return output

        # ========= TODO : END ========= #


class FeedForwardLayer(nn.Module):
    """
    Class definition for Feed Forward Layer.
    """

    def __init__(self, input_dim, feedforward_dim=None, dropout=0.1):
        """
        Initialize the Feed Forward Layer.

        The model should have the following layers:
        1. A linear layer for the feedforward network. (self.fc1) **set bias to True**
        2. A GELU activation function. (self.activation)
        3. A linear layer for the feedforward network. (self.fc2) ** set bias to True**
        4. A dropout layer. (self.dropout)

        NOTE : PLEASE KEEP OF EACH LAYER AS PROVIDED BELOW TO FACILITATE TESTING.
        """
        super().__init__()

        if feedforward_dim is None:
            feedforward_dim = input_dim * 4

        # ========= TODO : START ========= #

        self.fc1 = nn.Linear(input_dim, feedforward_dim, bias=True)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(feedforward_dim, input_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

        # ========= TODO : END ========= #

    def forward(self, x):
        """
        Forward pass of the Feed Forward Layer.

        Args:
        x : torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the input tokens.

        Output:
        torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the output tokens.
        """

        ### ========= TODO : START ========= ###

        return self.dropout(self.fc2(self.activation(self.fc1(x))))

        ### ========= TODO : END ========= ###


class LayerNorm(nn.Module):
    """
    LayerNorm module as in the paper https://arxiv.org/abs/1607.06450

    Note : Variance computation is done with biased variance.
    """

    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True) -> None:
        super().__init__()

        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(tuple(self.normalized_shape)))
            self.beta = nn.Parameter(torch.zeros(tuple(self.normalized_shape)))

    def forward(self, input):
        """
        Forward pass of the LayerNorm Layer.

        Args:
        input : torch.Tensor
            A tensor of shape (batch_size, ..., normalized_shape) containing the input.

        Output:
        torch.Tensor
            A tensor of shape (batch_size, ..., normalized_shape) containing the output.
        """
        
        ### ========= TODO : START ========= ###

        # Calculate mean and variance
        mean = input.mean(dim=-1, keepdim=True)
        variance = input.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize the input
        normalized_input = (input - mean) / torch.sqrt(variance + self.eps)

        # Apply learnable parameters if elementwise_affine is True
        if self.elementwise_affine:
            normalized_input = self.gamma * normalized_input + self.beta

        return normalized_input
        
        ### ========= TODO : END ========= ###



class TransformerLayer(nn.Module):
    """
    Class definition for a single transformer layer.
    """

    def __init__(self, input_dim, num_heads, feedforward_dim=None):
        super().__init__()
        """
        Initialize the Transformer Layer.
        We will use prenorm layer where we normalize the input before applying the attention and feedforward layers.

        The model should have the following layers:
        1. A LayerNorm layer. (self.norm1)
        2. A MultiHeadAttention layer. (self.attention)
        3. A LayerNorm layer. (self.norm2)
        4. A FeedForwardLayer layer. (self.feedforward)

        NOTE : PLEASE KEEP OF EACH LAYER AS PROVIDED BELOW TO FACILITATE TESTING.
        """

        # ========= TODO : START ========= #

        # self.norm1 = ...
        # self.attention = ...
        # self.norm2 = ...
        # self.feedforward = ...
        
        self.norm1 = LayerNorm(input_dim)
      
        self.attention = MultiHeadAttention(input_dim, num_heads, dropout=0.1)
   
        self.norm2 = LayerNorm(input_dim)
       
        self.feedforward = FeedForwardLayer(input_dim, feedforward_dim, dropout=0.1)
       
        

        # ========= TODO : END ========= #

    def forward(self, x):
        """
        Forward pass of the Transformer Layer.

        Args:
        x : torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the input tokens.

        Output:
        torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the output tokens.
        """

        # ========= TODO : START ========= #

        # Normalize and apply multi-head attention with residual connection
   
        x = x + self.attention(self.norm1(x))
        
        # Normalize and apply feedforward layer with residual connection
        x = x + self.feedforward(self.norm2(x))
        
        
        return x

        # ========= TODO : END ========= #


class MiniGPT(nn.Module):
    """
    Putting it all together: GPT model
    """

    def __init__(self, config) -> None:
        super().__init__()
        """
        Putting it all together: our own GPT model!

        Initialize the MiniGPT model.

        The model should have the following layers:
        1. An embedding layer that maps tokens to embeddings. (self.vocab_embedding)
        2. A positional embedding layer. (self.positional_embedding) We will use learnt positional embeddings. 
        3. A dropout layer for embeddings. (self.embed_dropout)
        4. Multiple TransformerLayer layers. (self.transformer_layers)
        5. A LayerNorm layer before the final layer. (self.prehead_norm)
        6. Final language Modelling head layer. (self.head) We will use weight tying (https://paperswithcode.com/method/weight-tying) and set the weights of the head layer to be the same as the vocab_embedding layer.

        NOTE: You do not need to modify anything here.
        """

        self.vocab_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.positional_embedding = nn.Embedding(
            config.context_length, config.embed_dim
        )
        self.embed_dropout = nn.Dropout(config.embed_dropout)

        self.transformer_layers = nn.ModuleList(
            [
                TransformerLayer(
                    config.embed_dim, config.num_heads, config.feedforward_size
                )
                for _ in range(config.num_layers)
            ]
        )

        # prehead layer norm
        self.prehead_norm = LayerNorm(config.embed_dim)

        self.head = nn.Linear(
            config.embed_dim, config.vocab_size
        )  # Language modelling head

        if config.weight_tie:
            self.head.weight = self.vocab_embedding.weight

        # precreate positional indices for the positional embedding
        pos = torch.arange(0, config.context_length, dtype=torch.long)
        self.register_buffer("pos", pos, persistent=False)

        self.apply(self._init_weights)

    def forward(self, x):
        """
        Forward pass of the MiniGPT model.

        Remember to add the positional embeddings to your input token!!

        Args:
        x : torch.Tensor
            A tensor of shape (batch_size, seq_len) containing the input tokens.

        Output:
        torch.Tensor
            A tensor of shape (batch_size, seq_len, vocab_size) containing the logits.
        """

        ### ========= TODO : START ========= ###

        # Get batch size and sequence length
        batch_size, seq_len = x.shape

        # Generate positional embeddings
        pos_embeddings = self.positional_embedding(self.pos[:seq_len])
        
        # Get token embeddings and add positional embeddings
        token_embeddings = self.vocab_embedding(x)
        embeddings = self.embed_dropout(token_embeddings + pos_embeddings)

        # Pass through each transformer layer
        for layer in self.transformer_layers:
            embeddings = layer(embeddings)

        # Apply final layer normalization
        embeddings = self.prehead_norm(embeddings)

        # Generate logits using the final linear layer
        logits = self.head(embeddings)

        return logits

        ### ========= TODO : END ========= ###

    def _init_weights(self, module):
        """
        Weight initialization for better convergence.

        NOTE : You do not need to modify this function.
        """

        if isinstance(module, nn.Linear):
            if module._get_name() == "fc2":
                # GPT-2 style FFN init
                torch.nn.init.normal_(
                    module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.num_layers)
                )
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def generate(self, context, max_new_tokens=100):
        """
        Use the model to generate new tokens given a context.

        Please copy the generate function from the BigramLanguageModel class you had implemented earlier.
        """

        ### ========= TODO : START ========= ###

        # context = torch.tensor(context, dtype=torch.long).unsqueeze(0)  # Convert context to tensor and add batch dimension
        # generated_tokens = context.tolist()[0]  # Initialize generated tokens with the context
        # new_tokens = []

        # for _ in range(max_new_tokens):
        #     logits = self.forward(context)  # Get logits from the model
        #     next_token_logits = logits[:, -1, :]  # Get logits for the last token in the sequence

        #     probabilities = torch.softmax(next_token_logits, dim=-1)  # Convert logits to probabilities
        #     next_token = torch.multinomial(probabilities, num_samples=1).item()  # Sample next token from multinomial distribution
        #     new_tokens.append(next_token)
        #     generated_tokens.append(next_token)  # Append the generated token to the sequence
        #     context = torch.tensor(generated_tokens, dtype=torch.long).unsqueeze(0)  # Update context with the new token

        # return next_token
        
        # context = torch.tensor(context, dtype=torch.long).unsqueeze(0)  # Convert context to tensor and add batch dimension
        # generated_tokens = context.tolist()[0]  # Initialize generated tokens with the context
    
        # for _ in range(max_new_tokens):
        #     logits = self.forward(context)  # Get logits from the model
        #     next_token_logits = logits[:, -1, :]  # Get logits for the last token in the sequence
    
        #     probabilities = torch.softmax(next_token_logits, dim=-1)  # Convert logits to probabilities
        #     next_token = torch.multinomial(probabilities, num_samples=1).item()  # Sample next token from multinomial distribution
        #     generated_tokens.append(next_token)  # Append the generated token to the sequence
        #     context = torch.tensor(generated_tokens, dtype=torch.long).unsqueeze(0)  # Update context with the new token
    
        # return generated_tokens
        # print("hey")
        
        context = torch.tensor(context, dtype=torch.long).unsqueeze(0).to(self.pos.device)  # Convert context to tensor, add batch dimension, and move to device
        generated_tokens = context.tolist()[0]  # Initialize generated tokens with the context
    
        for _ in range(max_new_tokens):
            logits = self.forward(context)  # Get logits from the model
            next_token_logits = logits[:, -1, :]  # Get logits for the last token in the sequence
    
            probabilities = torch.softmax(next_token_logits, dim=-1)  # Convert logits to probabilities
            next_token = torch.multinomial(probabilities, num_samples=1).item()  # Sample next token from multinomial distribution
            generated_tokens.append(next_token)  # Append the generated token to the sequence
            context = torch.tensor(generated_tokens, dtype=torch.long).unsqueeze(0).to(self.pos.device)  # Update context with the new token and move to device
    
        return generated_tokens

        ### ========= TODO : END ========= ###
