import torch
import torch.nn as nn

from .attack_torch import Attack


class FGSM(Attack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.FGSM(model, eps=8/255)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, eps=8/255):
        super().__init__("FGSM", model)
        self.eps = eps
        self.supported_mode = ['default', 'targeted']
        
        self.model=model

    def forward(self, images,labels,semi_targets,c,eta,eps):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # if self.targeted:
            # target_labels = self.get_target_label(images, labels)

        loss = nn.BCEWithLogitsLoss()

        images.requires_grad = True
        
        _,outputs=getScore(self.model,images,semi_targets,c,eta,eps)

        cost = +loss(outputs, labels.float())
        
        # outputs = self.get_logits(images)

        # # Calculate loss
        # if self.targeted:
        #     cost = -loss(outputs, target_labels)
        # else:
        #     cost = loss(outputs, labels)

        # Update adversarial images
        grad = torch.autograd.grad(cost, images,
                                   retain_graph=False, create_graph=False)[0]

        adv_images = images + self.eps*grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images
    
def getScore(net,inputs,semi_targets,c,eta,eps):
    outputs = net(inputs)
    dist = torch.sum((outputs - c) ** 2, dim=1)
    losses = torch.where(semi_targets == 0, dist, eta * ((dist + eps) ** semi_targets.float()))
    loss = torch.mean(losses)
    scores = dist
    
    return loss,scores
    