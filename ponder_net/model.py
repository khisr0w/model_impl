import torch
from torch import nn
import torch.nn.functional as F

class PonderNet(nn.Module):
    def __init__(self, in_dim: int, h_dim: int, max_steps: int):
        super().__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.max_steps = max_steps
        self.gru = nn.GRUCell(in_dim, h_dim)
        self.out = nn.Linear(h_dim, 1)
        self.halt = nn.Linear(h_dim, 1)
        self.should_halt = False

    def forward(self, x: torch.Tensor):
        b = x.shape[0]
        h = x.new_zeros((x.shape[0], self.h_dim))
        h = self.gru(x, h) # h_1

        p = [] # p_1 ... p_N
        y = [] # y_1 ... y_N

        one_minus_lambda_prod = h.new_ones((b,))

        batch_halted = h.new_zeros((b,)) # The halting status of each datapoint in the batch

        p_m = h.new_zeros((b,)) # Probabilities of the halted step
        y_m = h.new_zeros((b,)) # Predictions of the halted step

        for n in range(1, self.max_steps + 1):
            # TODO(Abid): Wouldn't it be better if one decide to halt or not at the end of the step?
            #             If we do it before, it essentially has to predict if the inner workings of
            #             the model, to predict, whereas at the end we simply look at the result to
            #             determine if problem had been solved or not.
            if n == self.max_steps:
                lambda_n = h.new_ones(h.shape[0])
            else:
                lambda_n = torch.sigmoid(self.halt(h))[:, 0] # P(n | ~n-1, ~n-2, ...)

            # NOTE(Abid): Total Probability of stopping at this iteration
            y_n = self.out(h)[:, 0]
            p_n = one_minus_lambda_prod * lambda_n

            # NOTE(Abid): Updating the Mul(1 - lambda) in case we don't halt in this step
            one_minus_lambda_prod = one_minus_lambda_prod * (1 - lambda_n)

            # NOTE(Abid): Sample halting given the probability
            batch_should_halt = torch.bernoulli(lambda_n) * (1 - batch_halted)

            p.append(p_n)
            y.append(y_n)
            # NOTE(Abid): Saving the probability (and prediction) of each datapoint at the step they halted.
            #             p_m and y_m only changes for those that are about to halt at this step.
            p_m = p_m * (1 - batch_should_halt) + p_n * batch_should_halt
            y_m = y_m * (1 - batch_should_halt) + y_n * batch_should_halt 
            batch_halted = batch_halted + batch_should_halt

            h = self.gru(x, h) # h_{t+1}

            if self.should_halt and batch_halted.sum() == b:
                break

        return torch.stack(p), torch.stack(y), p_m, y_m

class ReconstructionLoss(nn.Module):
    def __init__(self, loss_func):
        super().__init__()
        self.loss_func = loss_func

    def forward(self, p, y, y_grnd):
        total_loss = p.new_tensor(0.);

        N = p.shape[0]
        for n in range(N):
            # NOTE(Abid): In here, for some examples we use the loss of a step further from the step in which the datapoint was halted.
            #             Is this really desired? Isn't this a form of information leaking from across the batch? TODO: Perhaps a better
            #             option would be to only compute the expectaion of the loss until the datapoint halted (after which can be 0).
            #             As it currently is, if there is an example requiring a lot of steps, then the loss could increase for other
            #             datapoints simply due to the inclusion of that datapoint.
            loss = (p[n] * self.loss_func(y[n], y_grnd)).mean()
            total_loss = total_loss + loss

        return total_loss

class RegularizationLoss(nn.Module):
    def __init__(self, lambda_p, max_steps):
        super().__init__()
        p_g = torch.zeros((max_steps,))

        one_minus_lambda_prod = 1.
        for k in range(max_steps):
            p_g[k] = one_minus_lambda_prod * lambda_p
            one_minus_lambda_prod = one_minus_lambda_prod * (1 - lambda_p) # (1 - lambda_p)^k

        self.p_g = nn.Parameter(p_g, requires_grad=False)

    def forward(self, p):
        p = p.transpose(0, 1)
        p_g = self.p_g[None, :p.shape[1]].expand_as(p) # Take up until the last N entry in p, and then repeat (expand) across batch.

        # NOTE(Abid): In here we also influence the regularization of a datapoint based on how long the longest step was.
        #             Therefore, we allow other datapoints to influence this datapoint.
        return F.kl_div(p.log(), p_g, reduction="batchmean")

