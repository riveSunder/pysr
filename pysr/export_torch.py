torch_initialized = False
torch = None
sympytorch = None
PySRSymPyModule = None

def _initialize_torch():
    global torch_initialized
    global torch
    global sympytorch
    global PySRSymPyModule

    # Way to lazy load torch, only if this is called,
    # but still allow this module to be loaded in __init__
    if not torch_initialized:
        torch_initialized = True

        import torch as _torch
        import sympytorch as _sympytorch
        torch = _torch
        sympytorch = _sympytorch

        class _PySRSymPyModule(torch.nn.Module):
            def __init__(self, expression, symbols, extra_funcs=None):
                super().__init__()
                self._symbols = symbols
                self._expression = expression
                self._module = sympytorch.SymPyModule(
                        expressions=[expression],
                        extra_funcs=extra_funcs)

            def __repr__(self):
                return f"PySRTorchModule(X=>{self._expression})"

            def forward(self, X):
                symbols = {str(symbol): X[..., i]
                           for i, symbol in enumerate(self._symbols)}
                return self._module(**symbols)[..., 0]

        PySRSymPyModule = _PySRSymPyModule

def sympy2torch(expression, symbols_in, extra_torch_mappings=None):
    """Returns a module for a given sympy expression with trainable parameters;

    This function will assume the input to the module is a matrix X, where
        each column corresponds to each symbol you pass in `symbols_in`.
    """
    global PySRSymPyModule

    _initialize_torch()

    return PySRSymPyModule(expression, symbols_in, extra_funcs=extra_torch_mappings)
