class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.name = module
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()

# hookF = [Hook(layer[1]) for layer in list(self.net_g._modules.items())]
# hookB = [Hook(layer[1],backward=True) for layer in list(self.net_g._modules.items())]

# print('***'*3+'  Backward Hooks Inputs & Outputs  '+'***'*3)
# for hook in hookB:             
#     print(hook.name)
#     print('---'*17)
    
# assert(0)