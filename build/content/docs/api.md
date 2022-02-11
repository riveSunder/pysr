---
menu: main
title: API
---

<a id="pysr.sr"></a>

# pysr.sr

[[view_source]](https://github.com/MilesCranmer/PySR/blob/ec7b478418ea5be164495e3c94f46d0a7cae55a7/pysr/sr.py#L1)

<a id="pysr.sr.install"></a>

#### install

```python
def install(julia_project=None, quiet=False)
```

[[view_source]](https://github.com/MilesCranmer/PySR/blob/ec7b478418ea5be164495e3c94f46d0a7cae55a7/pysr/sr.py#L21)

Install PyCall.jl and all required dependencies for SymbolicRegression.jl.

Also updates the local Julia registry.

<a id="pysr.sr.PySRRegressor"></a>

## PySRRegressor Objects

```python
class PySRRegressor(BaseEstimator,  RegressorMixin)
```

[[view_source]](https://github.com/MilesCranmer/PySR/blob/ec7b478418ea5be164495e3c94f46d0a7cae55a7/pysr/sr.py#L361)

<a id="pysr.sr.PySRRegressor.__init__"></a>

#### \_\_init\_\_

```python
def __init__(model_selection="best", weights=None, binary_operators=None, unary_operators=None, procs=cpu_count(), loss="L2DistLoss()", populations=100, niterations=4, ncyclesperiteration=100, alpha=0.1, annealing=False, fractionReplaced=0.01, fractionReplacedHof=0.005, npop=100, parsimony=1e-4, migration=True, hofMigration=True, shouldOptimizeConstants=True, topn=10, weightAddNode=1, weightInsertNode=3, weightDeleteNode=3, weightDoNothing=1, weightMutateConstant=10, weightMutateOperator=1, weightRandomize=1, weightSimplify=0.002, perturbationFactor=1.0, extra_sympy_mappings=None, extra_torch_mappings=None, extra_jax_mappings=None, equation_file=None, verbosity=1e9, update_verbosity=None, progress=None, maxsize=20, fast_cycle=False, maxdepth=None, variable_names=None, batching=False, batchSize=50, select_k_features=None, warmupMaxsizeBy=0.0, constraints=None, useFrequency=True, tempdir=None, delete_tempfiles=True, julia_project=None, update=True, temp_equation_file=False, output_jax_format=False, output_torch_format=False, optimizer_algorithm="BFGS", optimizer_nrestarts=3, optimize_probability=1.0, optimizer_iterations=10, tournament_selection_n=10, tournament_selection_p=1.0, denoise=False, Xresampled=None, precision=32, multithreading=None, use_symbolic_utils=False, **kwargs, ,)
```

[[view_source]](https://github.com/MilesCranmer/PySR/blob/ec7b478418ea5be164495e3c94f46d0a7cae55a7/pysr/sr.py#L362)

Initialize settings for an equation search in PySR.

Note: most default parameters have been tuned over several example
equations, but you should adjust `niterations`,
`binary_operators`, `unary_operators` to your requirements.
You can view more detailed explanations of the options on the
[options page](https://pysr.readthedocs.io/en/latest/docs/options/) of the documentation.

**Arguments**:

- `model_selection` (`str`): How to select a model. Can be 'accuracy' or 'best'. The default, 'best', will optimize a combination of complexity and accuracy.
- `binary_operators` (`list`): List of strings giving the binary operators in Julia's Base. Default is ["+", "-", "*", "/",].
- `unary_operators` (`list`): Same but for operators taking a single scalar. Default is [].
- `niterations` (`int`): Number of iterations of the algorithm to run. The best equations are printed, and migrate between populations, at the end of each.
- `populations` (`int`): Number of populations running.
- `loss` (`str`): String of Julia code specifying the loss function.  Can either be a loss from LossFunctions.jl, or your own loss written as a function. Examples of custom written losses include: `myloss(x, y) = abs(x-y)` for non-weighted, or `myloss(x, y, w) = w*abs(x-y)` for weighted.  Among the included losses, these are as follows. Regression: `LPDistLoss{P}()`, `L1DistLoss()`, `L2DistLoss()` (mean square), `LogitDistLoss()`, `HuberLoss(d)`, `L1EpsilonInsLoss(ϵ)`, `L2EpsilonInsLoss(ϵ)`, `PeriodicLoss(c)`, `QuantileLoss(τ)`.  Classification: `ZeroOneLoss()`, `PerceptronLoss()`, `L1HingeLoss()`, `SmoothedL1HingeLoss(γ)`, `ModifiedHuberLoss()`, `L2MarginLoss()`, `ExpLoss()`, `SigmoidLoss()`, `DWDMarginLoss(q)`.
- `denoise` (`bool`): Whether to use a Gaussian Process to denoise the data before inputting to PySR. Can help PySR fit noisy data.
- `select_k_features` (`None/int`): whether to run feature selection in Python using random forests, before passing to the symbolic regression code. None means no feature selection; an int means select that many features.
- `procs` (`int`): Number of processes (=number of populations running).
- `multithreading` (`bool`): Use multithreading instead of distributed backend. Default is yes. Using procs=0 will turn off both.
- `batching` (`bool`): whether to compare population members on small batches during evolution. Still uses full dataset for comparing against hall of fame.
- `batchSize` (`int`): the amount of data to use if doing batching.
- `maxsize` (`int`): Max size of an equation.
- `ncyclesperiteration` (`int`): Number of total mutations to run, per 10 samples of the population, per iteration.
- `alpha` (`float`): Initial temperature.
- `annealing` (`bool`): Whether to use annealing. You should (and it is default).
- `fractionReplaced` (`float`): How much of population to replace with migrating equations from other populations.
- `fractionReplacedHof` (`float`): How much of population to replace with migrating equations from hall of fame.
- `npop` (`int`): Number of individuals in each population
- `parsimony` (`float`): Multiplicative factor for how much to punish complexity.
- `migration` (`bool`): Whether to migrate.
- `hofMigration` (`bool`): Whether to have the hall of fame migrate.
- `shouldOptimizeConstants` (`bool`): Whether to numerically optimize constants (Nelder-Mead/Newton) at the end of each iteration.
- `topn` (`int`): How many top individuals migrate from each population.
- `perturbationFactor` (`float`): Constants are perturbed by a max factor of (perturbationFactor*T + 1). Either multiplied by this or divided by this.
- `weightAddNode` (`float`): Relative likelihood for mutation to add a node
- `weightInsertNode` (`float`): Relative likelihood for mutation to insert a node
- `weightDeleteNode` (`float`): Relative likelihood for mutation to delete a node
- `weightDoNothing` (`float`): Relative likelihood for mutation to leave the individual
- `weightMutateConstant` (`float`): Relative likelihood for mutation to change the constant slightly in a random direction.
- `weightMutateOperator` (`float`): Relative likelihood for mutation to swap an operator.
- `weightRandomize` (`float`): Relative likelihood for mutation to completely delete and then randomly generate the equation
- `weightSimplify` (`float`): Relative likelihood for mutation to simplify constant parts by evaluation
- `equation_file` (`str`): Where to save the files (.csv separated by |)
- `verbosity` (`int`): What verbosity level to use. 0 means minimal print statements.
- `update_verbosity` (`int`): What verbosity level to use for package updates. Will take value of `verbosity` if not given.
- `progress` (`bool`): Whether to use a progress bar instead of printing to stdout.
- `maxdepth` (`int`): Max depth of an equation. You can use both maxsize and maxdepth.  maxdepth is by default set to = maxsize, which means that it is redundant.
- `fast_cycle` (`bool`): (experimental) - batch over population subsamples. This is a slightly different algorithm than regularized evolution, but does cycles 15% faster. May be algorithmically less efficient.
- `variable_names` (`list`): a list of names for the variables, other than "x0", "x1", etc.
- `warmupMaxsizeBy` (`float`): whether to slowly increase max size from a small number up to the maxsize (if greater than 0).  If greater than 0, says the fraction of training time at which the current maxsize will reach the user-passed maxsize.
- `constraints` (`dict`): dictionary of int (unary) or 2-tuples (binary), this enforces maxsize constraints on the individual arguments of operators. E.g., `'pow': (-1, 1)` says that power laws can have any complexity left argument, but only 1 complexity exponent. Use this to force more interpretable solutions.
- `useFrequency` (`bool`): whether to measure the frequency of complexities, and use that instead of parsimony to explore equation space. Will naturally find equations of all complexities.
- `tempdir` (`str/None`): directory for the temporary files
- `delete_tempfiles` (`bool`): whether to delete the temporary files after finishing
- `julia_project` (`str/None`): a Julia environment location containing a Project.toml (and potentially the source code for SymbolicRegression.jl).  Default gives the Python package directory, where a Project.toml file should be present from the install.
- `update` (`bool`): Whether to automatically update Julia packages.
- `temp_equation_file` (`bool`): Whether to put the hall of fame file in the temp directory. Deletion is then controlled with the delete_tempfiles argument.
- `output_jax_format` (`bool`): Whether to create a 'jax_format' column in the output, containing jax-callable functions and the default parameters in a jax array.
- `output_torch_format` (`bool`): Whether to create a 'torch_format' column in the output, containing a torch module with trainable parameters.
- `tournament_selection_n` (`int`): Number of expressions to consider in each tournament.
- `tournament_selection_p` (`float`): Probability of selecting the best expression in each tournament. The probability will decay as p*(1-p)^n for other expressions, sorted by loss.
- `precision` (`int`): What precision to use for the data. By default this is 32 (float32), but you can select 64 or 16 as well.
- `use_symbolic_utils` (`bool`): Whether to use SymbolicUtils during simplification.
- `**kwargs` (`dict`): Other options passed to SymbolicRegression.Options, for example, if you modify SymbolicRegression.jl to include additional arguments.

**Returns**:

Initialized model. Call `.fit(X, y)` to fit your data!

<a id="pysr.sr.PySRRegressor.__repr__"></a>

#### \_\_repr\_\_

```python
def __repr__()
```

[[view_source]](https://github.com/MilesCranmer/PySR/blob/ec7b478418ea5be164495e3c94f46d0a7cae55a7/pysr/sr.py#L712)

Prints all current equations fitted by the model.

The string `>>>>` denotes which equation is selected by the
`model_selection`.

<a id="pysr.sr.PySRRegressor.set_params"></a>

#### set\_params

```python
def set_params(**params)
```

[[view_source]](https://github.com/MilesCranmer/PySR/blob/ec7b478418ea5be164495e3c94f46d0a7cae55a7/pysr/sr.py#L763)

Set parameters for equation search.

<a id="pysr.sr.PySRRegressor.get_params"></a>

#### get\_params

```python
def get_params(deep=True)
```

[[view_source]](https://github.com/MilesCranmer/PySR/blob/ec7b478418ea5be164495e3c94f46d0a7cae55a7/pysr/sr.py#L773)

Get parameters for equation search.

<a id="pysr.sr.PySRRegressor.get_best"></a>

#### get\_best

```python
def get_best()
```

[[view_source]](https://github.com/MilesCranmer/PySR/blob/ec7b478418ea5be164495e3c94f46d0a7cae55a7/pysr/sr.py#L781)

Get best equation using `model_selection`.

<a id="pysr.sr.PySRRegressor.fit"></a>

#### fit

```python
def fit(X, y, weights=None, variable_names=None)
```

[[view_source]](https://github.com/MilesCranmer/PySR/blob/ec7b478418ea5be164495e3c94f46d0a7cae55a7/pysr/sr.py#L798)

Search for equations to fit the dataset and store them in `self.equations`.

**Arguments**:

- `X` (`np.ndarray/pandas.DataFrame`): 2D array. Rows are examples, columns are features. If pandas DataFrame, the columns are used for variable names (so make sure they don't contain spaces).
- `y` (`np.ndarray`): 1D array (rows are examples) or 2D array (rows are examples, columns are outputs). Putting in a 2D array will trigger a search for equations for each feature of y.
- `weights` (`np.ndarray`): Optional. Same shape as y. Each element is how to weight the mean-square-error loss for that particular element of y.
- `variable_names` (`list`): a list of names for the variables, other than "x0", "x1", etc.
You can also pass a pandas DataFrame for X.

<a id="pysr.sr.PySRRegressor.predict"></a>

#### predict

```python
def predict(X)
```

[[view_source]](https://github.com/MilesCranmer/PySR/blob/ec7b478418ea5be164495e3c94f46d0a7cae55a7/pysr/sr.py#L828)

Predict y from input X using the equation chosen by `model_selection`.

You may see what equation is used by printing this object. X should have the same
columns as the training data.

**Arguments**:

- `X` (`np.ndarray/pandas.DataFrame`): 2D array. Rows are examples, columns are features. If pandas DataFrame, the columns are used for variable names (so make sure they don't contain spaces).

**Returns**:

1D array (rows are examples) or 2D array (rows are examples, columns are outputs).

<a id="pysr.sr.PySRRegressor.sympy"></a>

#### sympy

```python
def sympy()
```

[[view_source]](https://github.com/MilesCranmer/PySR/blob/ec7b478418ea5be164495e3c94f46d0a7cae55a7/pysr/sr.py#L844)

Return sympy representation of the equation(s) chosen by `model_selection`.

<a id="pysr.sr.PySRRegressor.latex"></a>

#### latex

```python
def latex()
```

[[view_source]](https://github.com/MilesCranmer/PySR/blob/ec7b478418ea5be164495e3c94f46d0a7cae55a7/pysr/sr.py#L852)

Return latex representation of the equation(s) chosen by `model_selection`.

<a id="pysr.sr.PySRRegressor.jax"></a>

#### jax

```python
def jax()
```

[[view_source]](https://github.com/MilesCranmer/PySR/blob/ec7b478418ea5be164495e3c94f46d0a7cae55a7/pysr/sr.py#L860)

Return jax representation of the equation(s) chosen by `model_selection`.

Each equation (multiple given if there are multiple outputs) is a dictionary
containing {"callable": func, "parameters": params}. To call `func`, pass
func(X, params). This function is differentiable using `jax.grad`.

<a id="pysr.sr.PySRRegressor.pytorch"></a>

#### pytorch

```python
def pytorch()
```

[[view_source]](https://github.com/MilesCranmer/PySR/blob/ec7b478418ea5be164495e3c94f46d0a7cae55a7/pysr/sr.py#L880)

Return pytorch representation of the equation(s) chosen by `model_selection`.

Each equation (multiple given if there are multiple outputs) is a PyTorch module
containing the parameters as trainable attributes. You can use the module like
any other PyTorch module: `module(X)`, where `X` is a tensor with the same
column ordering as trained with.

