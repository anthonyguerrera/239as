"""
Training file for the models we implemented 
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils
from torch.utils.data import DataLoader
from einops import rearrange
import wandb

from model import BigramLanguageModel, MiniGPT
from dataset import TinyStoriesDataset
from config import BigramConfig, MiniGPTConfig


MODEL = "bigram"  # bigram or minigpt

if MODEL == "bigram":
    config = BigramConfig
    model = BigramLanguageModel(config)
elif MODEL == "minigpt":
    config = MiniGPTConfig
    model = MiniGPT(config)
else:
    raise ValueError("Invalid model name")


# Initialize wandb if you want to use it
if config.to_log:
    wandb.init()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


train_dataset = TinyStoriesDataset(
    config.path_to_data,
    mode="train",
    context_length=config.context_length,
)
eval_dataset = TinyStoriesDataset(
    config.path_to_data, mode="test", context_length=config.context_length
)

train_dataloader = DataLoader(
    train_dataset, batch_size=config.batch_size, pin_memory=True
)
eval_dataloader = DataLoader(
    eval_dataset, batch_size=config.batch_size, pin_memory=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("number of trainable parameters: %.2fM" % (count_parameters(model) / 1e6,))


if not Path.exists(config.save_path):
    Path.mkdir(MiniGPTConfig.save_path, parents=True, exist_ok=True)


### ==================== START OF YOUR CODE ==================== ###
"""
You are required to implement the training loop for the model.

Please keep the following in mind:
- You will need to define an appropriate loss function for the model.
- You will need to define an optimizer for the model.
- You are required to log the loss (either on wandb or any other logger you prefer) every `config.log_interval` iterations.
- It is recommended that you save the model weights every `config.save_iterations` iterations you can also just save the model with the best training loss.

Please check the config file to see the different configurations you can set for the model.
NOTE : 
The MiniGPT config has params that you do not need to use, these were added to scale the model but are 
not a required part of the assignment. 
Feel free to experiment with the parameters and I would be happy to talk to you about them if interested :)
"""
import torch.optim as optim
import os
import matplotlib.pyplot as plt

def train(model, train_dataloader, eval_dataloader, config):
    model.to(device)
    model.train()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Optionally initialize wandb logging
    if config.to_log:
        wandb.watch(model)

    # Lists to store loss values for plotting
    train_losses = []
    eval_losses = []
    #####
    
    #####

    # Training loop
    for epoch in range(config.max_iter):
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            ########
            if i >= 5000:
                break
            ########
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Log statistics
            running_loss += loss.item()
            if i % config.log_interval == (config.log_interval - 1):  # Log every log_interval mini-batches
                avg_loss = running_loss / config.log_interval
                print(f"[{epoch + 1}, {i + 1}] loss: {avg_loss:.3f}")
                train_losses.append(avg_loss)
                if config.to_log:
                    wandb.log({"loss": avg_loss})
                    print('loss:', avg_loss)
                running_loss = 0.0

            # Save the model every save_iterations
            if i % config.save_iterations == (config.save_iterations - 1):
                save_model(model, config.save_path, epoch, i)
                
        # Optionally evaluate the model after each epoch
        eval_loss = evaluate(model, eval_dataloader, config, criterion)
        eval_losses.append(eval_loss)

    print("Finished Training")
    save_model(model, config.save_path, epoch, i)

    # Plot the losses
    plot_losses(train_losses, eval_losses, config)

def evaluate(model, eval_dataloader, config, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data in eval_dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(eval_dataloader)
    print(f"Evaluation loss: {avg_loss:.3f}")
    if config.to_log:
        wandb.log({"eval_loss": avg_loss})
    model.train()
    return avg_loss

def save_model(model, path, epoch, iteration):
    save_path = os.path.join(path, f"model_epoch{epoch}_iter{iteration}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def plot_losses(train_losses, eval_losses, config):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(eval_losses, label='Evaluation Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training and Evaluation Loss')
    plt.legend()
    plt.savefig(os.path.join(config.save_path, 'loss_plot.png'))
    plt.show()

# Start the training process
train(model, train_dataloader, eval_dataloader, config)







# """
# Training file for the models we implemented
# """

# from pathlib import Path

# import torch
# import torch.nn as nn
# import torch.nn.utils
# from torch.utils.data import DataLoader
# from einops import rearrange
# import wandb
# import torch.optim as optim
# from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
# import matplotlib.pyplot as plt
# from model import BigramLanguageModel, MiniGPT
# from dataset import TinyStoriesDataset
# from config import BigramConfig, MiniGPTConfig


# MODEL = "minigpt"  # bigram or minigpt

# if MODEL == "bigram":
#     config = BigramConfig
#     model = BigramLanguageModel(config)
# elif MODEL == "minigpt":
#     config = MiniGPTConfig
#     model = MiniGPT(config)
# else:
#     raise ValueError("Invalid model name")


# # Initialize wandb if you want to use it
# if config.to_log:
#     wandb.init(project="dl2_proj3")


# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)


# train_dataset = TinyStoriesDataset(
#     config.path_to_data,
#     mode="train",
#     context_length=config.context_length,
# )
# eval_dataset = TinyStoriesDataset(
#     config.path_to_data, mode="test", context_length=config.context_length
# )

# train_dataloader = DataLoader(
#     train_dataset, batch_size=config.batch_size, pin_memory=True
# )
# eval_dataloader = DataLoader(
#     eval_dataset, batch_size=config.batch_size, pin_memory=True
# )

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# if torch.cuda.is_available():
#     print("Good to go!")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
#     print("using mps")
# else:
#     print("Please set GPU via the downward triangle in the top right corner.")

# print("number of trainable parameters: %.2fM" % (count_parameters(model) / 1e6,))


# if not Path.exists(config.save_path):
#     Path.mkdir(MiniGPTConfig.save_path, parents=True, exist_ok=True)


# ### ==================== START OF YOUR CODE ==================== ###
# """
# You are required to implement the training loop for the model.

# Please keep the following in mind:
# - You will need to define an appropriate loss function for the model.
# - You will need to define an optimizer for the model.
# - You are required to log the loss (either on wandb or any other logger you prefer) every `config.log_interval` iterations.
# - It is recommended that you save the model weights every `config.save_iterations` iterations you can also just save the model with the best training loss.

# Please check the config file to see the different configurations you can set for the model.
# NOTE :
# The MiniGPT config has params that you do not need to use, these were added to scale the model but are
# not a required part of the assignment.
# Feel free to experiment with the parameters and I would be happy to talk to you about them if interested :)
# """
# NUM_EPOCHS = 1
# LEARNING_RATE = 5e-4

# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,weight_decay=0.01)

# train_losses = []
# model.to(device)
# best_loss = float('inf')
# model.train()
# iteration_count = 0
# for epoch in range(NUM_EPOCHS):
#     for iteration, (inputs, targets) in enumerate(train_dataloader):
#         #early stopping
#         if iteration == 5000:
#             break
#         inputs, targets = inputs.to(device), targets.to(device)

#         outputs = model(inputs)
#         outputs = rearrange(outputs, 'b s v -> b s v')
#         #loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
#         loss = criterion(outputs.view(-1, config.vocab_size), targets.view(-1))

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#       # Add gradient clipping
#         if config.to_clip_grad:
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip)

#         # Add CosineAnnealingWarmRestarts lr scheduler
#         if config.scheduler:
#             scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=1)
#             scheduler.step()

#         if iteration % config.log_interval == 0:
#             train_losses.append((iteration, loss.item()))  # Log iteration and loss
#             if config.to_log:
#                 wandb.log({"loss": loss.item()})
#             print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{iteration+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")


#         if iteration % config.save_iterations == 0 or (iteration == len(train_dataloader) - 1):
#             if loss.item() < best_loss:
#                 best_loss = loss.item()
#                 torch.save(model.state_dict(), config.save_path / f"{MODEL}_best.pth")
#             torch.save(model.state_dict(), config.save_path / f"{MODEL}_last.pth")

# print("Training completed successfully.")


# iterations, losses = zip(*train_losses)  # Unzip iteration and loss
# plt.figure(figsize=(10, 5))
# plt.plot(iterations, losses, label='Training Loss')
# plt.xlabel('Iterations')
# plt.ylabel('Loss')
# plt.title('Training Loss over Iterations')
# plt.legend()
# plt.show()

