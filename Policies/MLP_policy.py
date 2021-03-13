from Policies.base_policy import BasePolicy
import torch
from torch import nn
from torch import optim
import utils.utils as utils
import utils.torch_utils as ptu

class MLPPolicy(BasePolicy, nn.Module):

    def __init__(self, args, **kwargs):
        super(BasePolicy, self).__init__(**kwargs)

        if args.discrete:
            self.logits_na = ptu.build_mlp(input_size=args.ob_dim,
                                      output_size=args.ac_dim,
                                      n_layers=args.n_layers,
                                      size=args.size)
            # self.logits_na.to(utils.device)
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        args.learning_rate)
        else:
            raise NotImplementedError
    
    def forward(self, ob):
        return self.logits_na(ob.float())

    def get_action(self, obs):

        if len(obs.shape) > 1:
            obs = obs
        else:
            obs = obs[None]
        obs = torch.from_numpy(obs)

        with torch.no_grad():
            probs = self(obs)
            distn = torch.distributions.Categorical(probs)
            action = distn.sample()

        action = action.squeeze(-1)
        return action.numpy()

    def update(self, obs, acs, advs):
        acs = torch.from_numpy(acs)
        obs = torch.from_numpy(obs)
        advs = torch.from_numpy(advs)

        probs = self(obs)
        distn = torch.distributions.Categorical(probs)

        loss = torch.sum(distn.log_prob(acs) * advs).mul(-1)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def save(self, filepath):
        raise NotImplementedError
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
    	raise NotImplementedError