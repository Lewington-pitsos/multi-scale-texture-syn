class Hook():
    features=None
    def __init__(self, layer): 
        self.hook = layer.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): 
        self.features = output
    def close(self): 
        self.hook.remove()


def attach_hooks(model, layer_indices):
  all_layers = list(model.children())

  return [Hook(all_layers[index]) for index in layer_indices]
