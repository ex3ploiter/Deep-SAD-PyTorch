import torch
import torch.nn as nn
import torch.optim as optim

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'



def fgsm(model, inputs, c, epsilon):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(inputs, requires_grad=True).to(device)
    
    
    outputs = model(inputs+delta)
    dist = torch.sum((outputs - c) ** 2, dim=1)
    scores=dist 
    
    scores.backward()

    # return inputs+epsilon * delta.grad.detach().sign()
    return epsilon * delta.grad.detach().sign()


def pgd(model, inputs, c, epsilon, alpha, num_iter):

    delta = torch.zeros_like(inputs, requires_grad=True).to(device)
    for t in range(num_iter):
        

        outputs = model(inputs+delta)
        dist = torch.sum((outputs - c) ** 2, dim=1)
        scores=dist 
    
        scores.backward()        
        
        
        
        delta.data = (delta + inputs.shape[0]*alpha*delta.grad.data).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    
    return delta.detach()


upper_limit, lower_limit = 1,0

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def pgd_inf( model, inputs, c, epsilon, alpha, num_iter,norm='l_inf'):
    delta = torch.zeros_like(inputs).to(device)
    if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
    delta = clamp(delta, lower_limit-inputs, upper_limit-inputs)
    delta.requires_grad = True
    for _ in range(num_iter):
        outputs = model(inputs+delta)
        dist = torch.sum((outputs - c) ** 2, dim=1)
        scores=dist 
    
        scores.backward()        
        
        if norm == "l_inf":
            delta.data = torch.clamp(delta + alpha * torch.sign(delta.grad.data), min=-epsilon, max=epsilon)
        delta.data = clamp(delta, lower_limit - inputs, upper_limit - inputs)
        delta.grad.zero_()
    
    return delta.detach()