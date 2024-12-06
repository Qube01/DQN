import torch
import torch.nn as nn
import torch.optim as optim

class ActionValueFunction(nn.Module):
    def __init__(self, state_size, action_space, learning_rate=0.01):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, len(action_space))
        )

        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def get_qvalue(self, fstate, action):
        fstate = torch.tensor(fstate, dtype=torch.float32)
        qvalues = self.model(fstate)
        return qvalues[action]

    def get_best_action(self, fstate, action_space):
        fstate = torch.tensor(fstate, dtype=torch.float32)
        qvalues = self.model(fstate)
        best_action_index = torch.argmax(qvalues).item()
        return action_space[best_action_index]

    def get_best_qvalue(self, fstate, action_space):
        fstate = torch.tensor(fstate, dtype=torch.float32)
        qvalues = self.model(fstate)
        best_qvalue = torch.max(qvalues).item()
        return best_qvalue

    def loss(self, targets, fstates, actions):
        fstates = torch.tensor(fstates, dtype=torch.float32)
        qvalues = self.model(fstates)
        qvalues = qvalues.gather(1, torch.tensor(actions).unsqueeze(1)).squeeze(1)
        targets = torch.tensor(targets, dtype=torch.float32)
        return self.loss_fn(qvalues, targets)

    def train_step(self, targets, fstates, actions):
        self.optimizer.zero_grad()
        loss_value = self.loss(targets, fstates, actions)
        loss_value.backward()
        nn.utils.clip_grad_value_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss_value.item()