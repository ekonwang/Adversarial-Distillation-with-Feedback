import torch
import torch.nn as nn
import torch.nn.functional as F

# batch mean reduction 
_KL_loss = nn.KLDivLoss(reduction='batchmean')
_XENT_loss = nn.CrossEntropyLoss()

# No mean reduction
_KL_no_reduction = nn.KLDivLoss(reduction='none')
_XENT_no_reduction = nn.CrossEntropyLoss(reduction='none')

class KLoss():
    def __init__(self, temp, no_reduction=False):
        self.temp = temp 
        self.no_reduction = no_reduction
    def __call__(self, basic_outputs, outputs):
        return KLoss.cal(basic_outputs, outputs, self.temp, self.no_reduction)
    @staticmethod 
    def cal(basic_outputs, outputs, temp=1., no_reduction=False):
        if no_reduction:
            return temp*temp*_KL_no_reduction(F.log_softmax(outputs/temp, dim=1),F.softmax(basic_outputs/temp, dim=1)).sum(dim=1)
        else:
            return temp*temp*_KL_loss(F.log_softmax(outputs/temp, dim=1),F.softmax(basic_outputs/temp, dim=1))


class XENTLoss():
    def __init__(self, no_reduction=False):
        self.no_reduction = no_reduction 
    def __call__(self, outputs, labels):
        return XENTLoss.cal(outputs, labels, self.no_reduction)
    @staticmethod
    def cal(outputs, labels, no_reduction=False):
        if no_reduction:
            return _XENT_no_reduction(outputs, labels)
        else:
            return _XENT_loss(outputs, labels)


class ARDLoss():
    def __init__(self, alpha, temp):
        self.alpha = alpha
        self.temp = temp 
    def __call__(self, basic_outputs, outputs, teacher_basic_outputs, targets):
        return ARDLoss.cal(basic_outputs, outputs, teacher_basic_outputs, targets, self.alpha, self.temp)
    @staticmethod 
    def cal(basic_outputs, outputs, teacher_basic_outputs, targets, alpha, temp):
        return alpha*KLoss.cal(teacher_basic_outputs, outputs, temp)+(1.0-alpha)*_XENT_loss(basic_outputs, targets)


class KDLoss():
    def __init__(self, alpha, temp):
        self.alpha = alpha
        self.temp = temp 
    def __call__(self, basic_outputs, teacher_basic_outputs, targets):
        return KDLoss.cal(basic_outputs, teacher_basic_outputs, targets, self.alpha, self.temp)
    @staticmethod 
    def cal(basic_outputs, teacher_basic_outputs, targets, alpha, temp):
        return alpha*KLoss.cal(teacher_basic_outputs, basic_outputs, temp)+(1.0-alpha)*_XENT_loss(basic_outputs, targets)

    
class ARDPROLoss():
    def __init__(self, alpha, temp):
        self.alpha = alpha
        self.temp = temp 
    def __call__(self, basic_outputs, outputs, teacher_basic_outputs, teacher_outputs, targets):
        return ARDPROLoss.cal(basic_outputs, outputs, teacher_basic_outputs, teacher_outputs, targets, self.alpha, self.temp)
    @staticmethod 
    def cal(basic_outputs, outputs, teacher_basic_outputs, teacher_outputs, targets, alpha, temp):
        stu_loss = ARDLoss.cal(basic_outputs, outputs, teacher_basic_outputs, targets, alpha, temp)
        teacher_loss = XENTLoss.cal(teacher_outputs, targets)
        return stu_loss, teacher_loss


class KLCoarseLoss():
    def __init__(self, alpha, temp):
        self.alpha = alpha
        self.temp = temp 
    def __call__(self, basic_outputs, outputs, teacher_basic_outputs, teacher_outputs, targets):
        return ARDPROLoss.cal(basic_outputs, outputs, teacher_basic_outputs, teacher_outputs, targets, self.alpha, self.temp)
    @staticmethod 
    def cal(basic_outputs, outputs, teacher_basic_outputs, teacher_outputs, targets, alpha, temp):
        stu_loss = ARDLoss.cal(basic_outputs, outputs, teacher_basic_outputs, targets, alpha, temp)
        teacher_loss = KLoss.cal(teacher_basic_outputs, teacher_outputs, temp)
        return stu_loss, teacher_loss


class TRADESLoss():
    def __init__(self, temp, lamda):
        self.temp = temp
        self.lamda = lamda 
    def __call__(self, basic_outputs, outputs, targets):
        return TRADESLoss.cal(basic_outputs, outputs, targets, self.temp, self.lamda)
    @staticmethod 
    def cal(basic_outputs, outputs, targets, temp, lambd):
        sat_loss = XENTLoss.cal(basic_outputs, targets)
        kl_loss = KLoss.cal(basic_outputs, outputs, temp)*lambd
        return sat_loss+kl_loss


class SELoss():
    def __init__(self):
        pass
    def __call__(self, basic_outputs, outputs):
        return SELoss.cal(basic_outputs, outputs)
    @staticmethod 
    def cal(basic_outputs, outputs, no_reduction=False):
        score = torch.sum((F.softmax(basic_outputs, dim=1) - F.softmax(outputs, dim=1))**2, dim=1)
        return (score if no_reduction else score.mean())

# 不同的 Alpha 因子算法可能有强度不一致的问题
# 解决强度不一致或许不能简单地将 mean alpha 置为 1，而需要将 max alpha 置 1，同时保证总体强度（通过次方运算）


def rank(values):
    # 保证均值为 1
    return torch.argsort(torch.argsort(values)).to(torch.float) / (values.shape[0]-1) * 2

class AlphaFactor():
    def __init__(self, factor=1.):
        self.factor = factor


class AlphaFactorLeast(AlphaFactor):
    adjust_const = 1/0.7301
    def __init__(self, factor=1.):
        super().__init__(factor)

    def __call__(self, outputs, targets):
        return AlphaFactorLeast.cal(outputs, targets, self.factor)
    
    @staticmethod 
    def cal(outputs, targets, factor=1.):
        values = (1-AlphaFactorMost.cal(outputs, targets)**factor)
        return values


class AlphaFactorMost(AlphaFactor):
    adjust_const = 1/0.2813
    def __init__(self, factor=1.):
        super().__init__(factor)

    def __call__(self, outputs, targets):
        return AlphaFactorMost.cal(outputs, targets, self.factor)
    
    @staticmethod 
    def cal(outputs, targets, factor=1.):
        values = F.softmax(outputs, dim=1)[torch.arange(outputs.shape[0]), targets]**factor
        return values

class AlphaFactorTargetSE(AlphaFactor):
    adjust_const = 1/0.1514
    def __init__(self, factor=1.):
        super().__init__(factor)

    def __call__(self, basic_outputs, outputs, targets):
        return AlphaFactorTargetSE.cal(basic_outputs, outputs, targets)
    
    @staticmethod 
    def cal(basic_outputs, outputs, targets):
        basic_out = AlphaFactorMost.cal(basic_outputs, targets)
        out = AlphaFactorMost.cal(outputs, targets)
        values = (basic_out - out)**2
        return values


class AlphaFactorSE(AlphaFactor):
    adjust_const = 1/0.2341
    def __init__(self, factor=1.):
        super().__init__(factor)

    def __call__(self, basic_outputs, outputs):
        return AlphaFactorSE.cal(basic_outputs, outputs)
    
    @staticmethod 
    def cal(basic_outputs, outputs):
        values = SELoss.cal(basic_outputs, outputs)
        return values


def fosc_deps(pert_inputs, teacher_outputs, targets):
    xent_loss = XENTLoss.cal(teacher_outputs, targets)
    grad = torch.autograd.grad(xent_loss, [pert_inputs], retain_graph=True)[0]
    return grad.detach()

def dot(vec1, vec2):
    return torch.sum(vec1 * vec2, dim=(1, 2, 3))


class AlphaFOSC(AlphaFactor):
    def __init__(self, factor=1.):
        super().__init__(factor)

    def __call__(self, epsilon, grad, pert_inputs, origin_inputs):
        return AlphaFOSC.cal(epsilon, grad, pert_inputs, origin_inputs)
    
    @staticmethod
    def cal(epsilon, grad, pert_inputs, origin_inputs):
        return epsilon*torch.norm(grad, p=1, dim=(1, 2, 3)) - dot(pert_inputs-origin_inputs, grad)


class AlphainvFOSC(AlphaFactor):
    def __init__(self, factor=1.):
        super().__init__(factor)

    def __call__(self, epsilon, grad, pert_inputs, origin_inputs):
        return AlphaFOSC.cal(epsilon, grad, pert_inputs, origin_inputs)
    
    @staticmethod
    def cal(epsilon, grad, pert_inputs, origin_inputs):
        fosc = AlphaFOSC.cal(epsilon, grad, pert_inputs, origin_inputs)
        max_fosc, min_fosc = torch.max(fosc), torch.min(fosc)
        return max_fosc + min_fosc - fosc
