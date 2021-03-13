import torch
from torch import nn
from torch import optim
from Critics.base_critic import BaseCritic
import utils.torch_utils as ptu

class VanillaCritic(nn.Module, BaseCritic):
    
    def __init__(self, args):
        super().__init__()

        self.gamma = args.gamma
        self.critic_network = ptu.build_mlp(
            args.ob_dim,
            1,
            n_layers=args.n_layers,
            size=args.size,
        )
        # self.critic_network.to(ptu.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.critic_network.parameters(),
            args.learning_rate,
        )
    
    def critic_prediction(self, obs):
        if len(obs.shape) > 1:
            obs = obs
        else:
            obs = obs[None]
        obs = torch.from_numpy(obs)

        with torch.no_grad():
            v = self(obs)

        v = v.squeeze(-1)
        return v
    
    def forward(self, input):
        return self.critic_network(input.float())

    def update(self, ob_no, next_ob_no, re_n, terminal_n):
        terminal_n = torch.from_numpy(terminal_n)
        re_n = torch.from_numpy(re_n)

        target = re_n + self.gamma * self.critic_prediction(next_ob_no) * torch.logical_not(terminal_n)
        
        ob_no = torch.from_numpy(ob_no)
        loss = self.criterion(self(ob_no.float()), target.float())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
        

        