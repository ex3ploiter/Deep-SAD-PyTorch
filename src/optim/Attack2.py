

import torch
import torch.nn as nn
import torch.optim as optim

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

upper_limit = 1
lower_limit = 0


def fgsm(model, inputs, c, epsilon):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(inputs, requires_grad=True).to(device)
    
    
    outputs = model(inputs+delta)
    dist = torch.sum((outputs - c) ** 2, dim=1)
    scores=dist 
    
    scores.backward()

    # return inputs+epsilon * delta.grad.detach().sign()
    return epsilon * delta.grad.detach().sign()


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def pgd_inf(model, X, epsilon, alpha, attack_iters, restarts,c):
    
    print("\n\nthis is eps:   ",epsilon,"\n\n")
    
    max_loss = torch.zeros(X.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)


            
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            dist = torch.sum((output - c) ** 2, dim=1)
            scores=dist 
            scores.backward()        
            
            
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        
        # all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
        all_loss = getScore(model,X,delta,c)
        
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta

def attack_pgd(model, X, epsilon=8/255, alpha=2/255, attack_iters=10, restarts=1, norm="l_inf",c=None):
    max_loss = torch.zeros(X.shape[0]).to(device)
    max_delta = torch.zeros_like(X).to(device)
    for _ in range(restarts):
        delta = torch.zeros_like(X).to(device)
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)

        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            # output = forward(X + delta).view(-1)
            index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            loss = getScore(model,X,delta,c)
            loss.backward()
            
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()

        
        all_loss = getScore(model,X,delta,c)
        
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta.detach()


def getScore(model,X,delta,c):
    output = model(X + delta)
    dist = torch.sum((output - c) ** 2, dim=1)
    scores=dist 
    return scores
    