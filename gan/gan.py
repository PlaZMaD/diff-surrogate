import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch


class PsiCompressor(nn.Module):
    def __init__(self, input_param, out_dim):
        super().__init__()
        self.fc = nn.Linear(input_param, out_dim)
        self.out_dim = out_dim
        
    def forward(self, psi):
        return torch.tanh(self.fc(psi))


class Generator(nn.Module):
    def __init__(self, noise_dim, out_dim, psi_compressor, hidden_dim=100, X_dim=1, psi_dim=2):
        super(Generator, self).__init__()
        
        self.fc1 = nn.Linear(noise_dim + X_dim + psi_dim, hidden_dim)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)
        
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.constant_(self.fc3.bias, 0.0)
        
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_normal_(self.fc4.weight)
        nn.init.constant_(self.fc4.bias, 0.0)        
        
        self.pc = psi_compressor
        self.psi_dim = psi_dim

    def forward(self, z, params):
        """
            Generator takes a vector of noise and produces sample
        """
        #psi_embedding = self.pc(params[:, :self.psi_dim])
        #z = torch.cat([z, psi_embedding, params[:, self.psi_dim:]], dim=1)
        z = torch.cat([z, params], dim=1)
        h1 = torch.tanh(self.fc1(z))
        h4 = torch.tanh(self.fc4(h1))
        h2 = F.leaky_relu(self.fc2(h4))
        y_gen = self.fc3(h2)
        return y_gen
    
class Discriminator(nn.Module):
    def __init__(self, in_dim, psi_compressor, hidden_dim=100, X_dim=1, psi_dim=2):
        super(Discriminator, self).__init__()
        
        self.fc1 = nn.Linear(in_dim + X_dim + psi_compressor.out_dim, hidden_dim)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)
        
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.constant_(self.fc3.bias, 0.0)
        
        self.fc4 = nn.Linear(hidden_dim, 1)
        nn.init.xavier_normal_(self.fc4.weight)
        nn.init.constant_(self.fc4.bias, 0.0)
        
        self.pc = psi_compressor
        self.psi_dim = psi_dim

    def forward(self, x, params):
        #x = torch.cat([x, params], dim=1)
        psi_embedding = self.pc(params[:, :self.psi_dim])
        x = torch.cat([x, psi_embedding, params[:, self.psi_dim:]], dim=1)
        h1 = torch.tanh(self.fc1(x))
        h2 = F.leaky_relu(self.fc2(h1))
        #h3 = F.leaky_relu(self.fc3(h2))
        score = torch.sigmoid(self.fc4(h2))
        return score
    
class WSDiscriminator(nn.Module):
    def __init__(self, in_dim, psi_compressor, hidden_dim=100, X_dim=1, psi_dim=2):
        super().__init__()

        self.fc1 = nn.Linear(in_dim + X_dim + psi_compressor.out_dim, hidden_dim)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.constant_(self.fc3.bias, 0.0)

        self.fc4 = nn.Linear(hidden_dim, 1)
        nn.init.xavier_normal_(self.fc4.weight)
        nn.init.constant_(self.fc4.bias, 0.0)

        self.pc = psi_compressor
        self.psi_dim = psi_dim        
        
    def forward(self, x, params):
        #x = torch.cat([x, params], dim=1)
        psi_embedding = self.pc(params[:, :self.psi_dim])
        x = torch.cat([x, psi_embedding, params[:, self.psi_dim:]], dim=1)        
        h1 = torch.tanh(self.fc1(x))
        h2 = F.leaky_relu(self.fc2(h1))
        #h3 = F.leaky_relu(self.fc3(h2))
        score = self.fc4(h2)
        return score    
    
    
class GANLosses(object):
    def __init__(self, task, device):
        self.TASK = task
        self.device = device
    
    def g_loss(self, discrim_output):
        eps = 1e-10
        if self.TASK == 1: 
            loss = torch.log(1 - discrim_output + eps).mean()    
        elif self.TASK in [2,5]:
            loss = - torch.log(discrim_output + eps).mean()
        elif self.TASK in (3, 4):
            loss = - discrim_output.mean()
        return loss

    def d_loss(self, discrim_output_gen, discrim_output_real):
        eps = 1e-10
        if self.TASK in (1, 2, 5): 
            loss = - torch.log(discrim_output_real + eps).mean() - torch.log(1 - discrim_output_gen + eps).mean()
        elif self.TASK in (3, 4):
            loss = - (discrim_output_real.mean() - discrim_output_gen.mean())
        return loss

    def calc_gradient_penalty(self, discriminator, data_gen, inputs_batch, inp_data, lambda_reg = .1):
        alpha = torch.rand(inp_data.shape[0], 1)
        alpha = alpha.expand(inp_data.size()).to(self.device)

        interpolates = (alpha * inp_data + ((1 - alpha) * data_gen)).to(self.device)

        interpolates.requires_grad = True

        disc_interpolates = discriminator(interpolates, inputs_batch)

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_reg
        return gradient_penalty
    
    def calc_zero_centered_GP(self, discriminator, data_gen, inputs_batch, inp_data, gamma_reg = .1):
        
        local_input = inp_data.clone().detach().requires_grad_(True)
        disc_interpolates = discriminator(local_input, inputs_batch)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=local_input,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        return gamma_reg / 2 * (gradients.norm(2, dim=1) ** 2).mean() 
