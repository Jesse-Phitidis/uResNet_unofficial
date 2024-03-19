from torch.optim.lr_scheduler import LambdaLR

class uResNet_Scheduler(LambdaLR):
    def __init__(self, optimizer):
        func = lambda epoch: 0.1 if epoch >= 415 else 1
        super().__init__(optimizer, func)
    