from collections import OrderedDict
import torch
from torch import nn
from torch import optim
from Critics.base_critic import BaseCritic
import utils.torch_utils as ptu
import numpy as np

class VanillaCritic(nn.Module, BaseCritic):
    
    def __init__(self, args):
        super().__init__()

        self.gamma = args.gamma
        self.critic_network = ptu.build_mlp(
            args.ob_dim,
            1,
            n_layers=args.n_layers,
            size=args.size,
            output_activation_str='identity'
        )
        print(self.critic_network)
        # self.critic_network.to(ptu.device)
        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer = optim.Adam(
            self.critic_network.parameters(),
            lr=args.learning_rate,
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
        return self.critic_network(input)

    def update(self, ob_no, next_ob_no, re_n, terminal_n):
        assert ob_no.shape[0] == next_ob_no.shape[0] == re_n.shape[0] == terminal_n.shape[0]
        terminal_n = torch.from_numpy(terminal_n)
        re_n = torch.from_numpy(re_n)
        ob_no = torch.from_numpy(ob_no)

        for _ in range(10):
            target = re_n + self.gamma * self.critic_prediction(next_ob_no) * torch.logical_not(terminal_n)
            for _ in range(10):
                pred = torch.squeeze(self(ob_no))
                loss = self.criterion(pred, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        log = OrderedDict()
        predictions = self.critic_prediction(ob_no.numpy()).numpy()
        # print(predictions)
        log["Mean_CriticPredict"] = np.mean(predictions)
        return log
        

        