import torch
import torch.nn as nn
import torch.nn.functional as F
from ResUNet import ConditionalUnet
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConditionalDDPM(nn.Module):
    def __init__(self, dmconfig):
        super().__init__()
        self.dmconfig = dmconfig
        self.loss_fn = nn.MSELoss()
        self.network = ConditionalUnet(1, self.dmconfig.num_feat, self.dmconfig.num_classes)
    def scheduler(self, t_s):
        beta_1, beta_T, T = self.dmconfig.beta_1, self.dmconfig.beta_T, self.dmconfig.T
    
        # Precompute the entire schedule arrays
        all_timesteps = torch.arange(1, T + 1, dtype=torch.float32, device=t_s.device)
        beta_t_all = beta_1 + (beta_T - beta_1) * (all_timesteps - 1) / (T - 1)
        alpha_t_all = 1 - beta_t_all
        alpha_t_bar_all = torch.cumprod(alpha_t_all, dim=0)
    
        # Initialize tensors to store per-batch-item schedule results
        beta_t, sqrt_beta_t, alpha_t = None, None, None
        oneover_sqrt_alpha, alpha_t_bar, sqrt_alpha_bar, sqrt_oneminus_alpha_bar=None, None, None,None
    
        # Loop over each batch element
        if t_s.ndim:
            beta_t = torch.empty_like(t_s, dtype=torch.float32)
            sqrt_beta_t = torch.empty_like(t_s, dtype=torch.float32)
            alpha_t = torch.empty_like(t_s, dtype=torch.float32)
            oneover_sqrt_alpha = torch.empty_like(t_s, dtype=torch.float32)
            alpha_t_bar = torch.empty_like(t_s, dtype=torch.float32)
            sqrt_alpha_bar = torch.empty_like(t_s, dtype=torch.float32)
            sqrt_oneminus_alpha_bar = torch.empty_like(t_s, dtype=torch.float32)
            
            for i in range(t_s.size(0)):
                index = int(t_s[i].item() - 1)  # Convert to zero-index
                beta_t[i] = beta_t_all[index]
                sqrt_beta_t[i] = torch.sqrt(beta_t[i])
                alpha_t[i] = alpha_t_all[index]
                oneover_sqrt_alpha[i] = torch.sqrt(1.0 / alpha_t[i])
                alpha_t_bar[i] = alpha_t_bar_all[index]
                sqrt_alpha_bar[i] = torch.sqrt(alpha_t_bar[i])
                sqrt_oneminus_alpha_bar[i] = torch.sqrt(1 - alpha_t_bar[i])
        else:
            beta_t = beta_t_all[t_s - 1]  # Adjust index since t_s is 1-indexed
            sqrt_beta_t = torch.sqrt(beta_t)
            alpha_t = alpha_t_all[t_s - 1]
            oneover_sqrt_alpha = torch.sqrt(1.0 / alpha_t)
            alpha_t_bar = alpha_t_bar_all[t_s - 1]
            sqrt_alpha_bar = torch.sqrt(alpha_t_bar)
            sqrt_oneminus_alpha_bar = torch.sqrt(1 - alpha_t_bar)
            
    
        return {
            'beta_t': beta_t,
            'sqrt_beta_t': sqrt_beta_t,
            'alpha_t': alpha_t,
            'sqrt_alpha_bar': sqrt_alpha_bar,
            'oneover_sqrt_alpha': oneover_sqrt_alpha,
            'alpha_t_bar': alpha_t_bar,
            'sqrt_oneminus_alpha_bar': sqrt_oneminus_alpha_bar
        }

    
    # def scheduler(self, t_s):
    #     beta_1, beta_T, T = self.dmconfig.beta_1, self.dmconfig.beta_T, self.dmconfig.T
    #     # ==================================================== #
    #     # YOUR CODE HERE:
    #     #   Inputs:
    #     #       t_s: the input time steps, with shape (B,1). 
    #     #   Outputs:
    #     #       one dictionary containing the variance schedule
    #     #       $\beta_t$ along with other potentially useful constants.  
        
    #     # t_steps = torch.arange(1, T + 1, dtype=torch.float32) / T

    #     all_timesteps = torch.arange(1, T + 1, dtype=torch.float32).to(t_s.device)  # Ensure same device as t_s
    #     beta_t_all = beta_1 + (beta_T - beta_1) * (all_timesteps - 1) / (T - 1)  # Interpolate for all steps

    #     # Now calculate alpha_t for all time steps
    #     alpha_t_all = 1 - beta_t_all

    #     # Compute cumulative product of alpha_t for all time steps to get alpha_t_bar
    #     alpha_t_bar_all = torch.cumprod(alpha_t_all, dim=0)

    #     # Extract the results for the specific time step(s) given in t_s
    #     # This assumes t_s are 1-indexed and directly correspond to indices in the computed tensors
    #     beta_t = beta_t_all[t_s - 1]  # Adjust index since t_s is 1-indexed
    #     sqrt_beta_t = torch.sqrt(beta_t)
    #     alpha_t = alpha_t_all[t_s - 1]
    #     oneover_sqrt_alpha = torch.sqrt(1.0 / alpha_t)
    #     alpha_t_bar = alpha_t_bar_all[t_s - 1]
    #     sqrt_alpha_bar = torch.sqrt(alpha_t_bar)
    #     sqrt_oneminus_alpha_bar = torch.sqrt(1 - alpha_t_bar)

    #     print("scheduler working")

    #     # ==================================================== #
    #     return {
    #         'beta_t': beta_t,
    #         'sqrt_beta_t': sqrt_beta_t,
    #         'alpha_t': alpha_t,
    #         'sqrt_alpha_bar': sqrt_alpha_bar,
    #         'oneover_sqrt_alpha': oneover_sqrt_alpha,
    #         'alpha_t_bar': alpha_t_bar,
    #         'sqrt_oneminus_alpha_bar': sqrt_oneminus_alpha_bar
    #     }

    def forward(self, images, conditions):
        T = self.dmconfig.T
        noise_loss = None
        # ==================================================== #
        # YOUR CODE HERE:
        #   Complete the training forward process based on the
        #   given training algorithm.
        #   Inputs:
        #       images: real images from the dataset, with size (B,1,28,28).
        #       conditions: condition labels, with size (B). You should
        #                   convert it to one-hot encoded labels with size (B,10)
        #                   before making it as the input of the denoising network.
        #   Outputs:
        #       noise_loss: loss computed by the self.loss_fn function  
        B = images.size()[0]

        conditions = F.one_hot(conditions, num_classes=10).float()

        # Generate noise
        noise = torch.randn_like(images)
    
        # Sample time steps uniformly
        # print("step0")
        t = torch.randint(1, T, (B, 1), dtype=torch.float32, device=images.device)
        # print("step1")
        schedule = self.scheduler(t)
        # print("step2")
        # print(B)
        # print(schedule['sqrt_alpha_bar'].shape)
        sqrt_alpha_bar = schedule['sqrt_alpha_bar'].reshape(B, 1, 1, 1)
        sqrt_oneminus_alpha_bar = schedule['sqrt_oneminus_alpha_bar'].reshape(B, 1, 1, 1)
        # print(images.shape, noise.shape)
        # Corrupt the images
        x_t = sqrt_alpha_bar * images + sqrt_oneminus_alpha_bar * noise
    
        # Normalize and adjust t tensor shape correctly
        t_normalized = t / T
        # Ensure t is correctly shaped as a single dimensional float tensor per batch element
        t_normalized = t_normalized.view(B, 1)  # Assuming ConditionalUnet expects B x 1 for t
    
        # Forward pass through ConditionalUnet
        # print("")
        # print(x_t.shape, conditions.shape, t_normalized.shape)
        unet_out = self.network(x_t, t_normalized, conditions)
    
        # Calculate loss
        noise_loss = self.loss_fn(unet_out, noise)

        # ==================================================== #
        
        return noise_loss

    def sample(self, conditions, omega):
        T = self.dmconfig.T
        X_t = None
        # ==================================================== #
        # YOUR CODE HERE:
        #   Complete the training forward process based on the
        #   given sampling algorithm.
        #   Inputs:
        #       conditions: condition labels, with size (B). You should
        #                   convert it to one-hot encoded labels with size (B,10)
        #                   before making it as the input of the denoising network.
        #       omega: conditional guidance weight.
        #   Outputs:
        #       generated_images
        condition_mask_value = self.dmconfig.condition_mask_value
        device = next(self.network.parameters()).device
        B = conditions.shape[0]
        input_dim = self.dmconfig.input_dim
        num_channels = self.dmconfig.num_channels

        # start from a completely gaussian noise image
        X_t = torch.randn(B,num_channels,input_dim[0],input_dim[1], device=device)

        with torch.no_grad():
            for t in torch.arange(T,0,-1):
                tensor_t = torch.full((B,num_channels,1,1),t, device=device)
                # sample unit variance guassian noise
                Z = torch.randn_like(X_t, device=device) if t > 1 else torch.zeros_like(X_t, device=device)

                # grab scheduler parameters
                schedule_dict = self.scheduler(t)
                beta_t = schedule_dict['beta_t'].view(-1,1,1,1).to(device)
                alpha_t = schedule_dict['alpha_t'].view(-1,1,1,1).to(device)
                oneover_sqrt_alpha = schedule_dict['oneover_sqrt_alpha'].view(-1,1,1,1).to(device)
                sqrt_oneminus_alpha_bar = schedule_dict['sqrt_oneminus_alpha_bar'].view(-1,1,1,1).to(device)
                sigma_t = torch.sqrt(beta_t)

                # normalize time step t to range 0-1 (improves output stability)
                tensor_t = (tensor_t/T).view(-1,1,1,1)
                # conditional prediction for timestep t
                cond_predict = self.network(X_t, tensor_t, conditions)
                # unconditional prediction for timestep t
                uncond_predict = self.network(X_t, tensor_t, conditions*condition_mask_value)
                # calculate corrected noise
                epsilon_t = (1 + omega)*cond_predict - omega*uncond_predict

                # update X_t
                X_t = oneover_sqrt_alpha * (X_t - (1-alpha_t)/sqrt_oneminus_alpha_bar*epsilon_t) + sigma_t * Z

        


        # ==================================================== #
        generated_images = (X_t * 0.3081 + 0.1307).clamp(0,1) # denormalize the output images
        return generated_images