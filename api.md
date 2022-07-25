
<a id="pysr.sr"></a>

# pysr.sr

[[view_source]](https://github.com/MilesCranmer/PySR/blob/8da5000dfc58c1a035c623ebd6c6e2acea472134/pysr/sr.py#L1)

<a id="pysr.sr.PySRRegressor"></a>

## PySRRegressor Objects

```python
class PySRRegressor(MultiOutputMixin,  RegressorMixin,  BaseEstimator)
```

[[view_source]](https://github.com/MilesCranmer/PySR/blob/8da5000dfc58c1a035c623ebd6c6e2acea472134/pysr/sr.py#L194)

High-performance symbolic regression.

This is the scikit-learn interface for SymbolicRegression.jl.
This model will automatically search for equations which fit
a given dataset subject to a particular loss and set of
constraints.

Parameters
----------
model_selection : str, default="best"
    Model selection criterion. Can be 'accuracy' or 'best'.
    `"accuracy"` selects the candidate model with the lowest loss
    (highest accuracy). `"best"` selects the candidate model with
    the lowest sum of normalized loss and complexity.

binary_operators : list[str], default=["+", "-", "*", "/"]
    List of strings giving the binary operators in Julia's Base.

unary_operators : list[str], default=None
    Same as :param`binary_operators` but for operators taking a
    single scalar.

niterations : int, default=40
    Number of iterations of the algorithm to run. The best
    equations are printed and migrate between populations at the
    end of each iteration.

populations : int, default=15
    Number of populations running.

population_size : int, default=33
    Number of individuals in each population.

max_evals : int, default=None
    Limits the total number of evaluations of expressions to
    this number.

maxsize : int, default=20
    Max complexity of an equation.

maxdepth : int, default=None
    Max depth of an equation. You can use both :param`maxsize` and
    :param`maxdepth`. :param`maxdepth` is by default not used.

warmup_maxsize_by : float, default=0.0
    Whether to slowly increase max size from a small number up to
    the maxsize (if greater than 0).  If greater than 0, says the
    fraction of training time at which the current maxsize will
    reach the user-passed maxsize.

timeout_in_seconds : float, default=None
    Make the search return early once this many seconds have passed.

constraints : dict[str, int | tuple[int,int]], default=None
    Dictionary of int (unary) or 2-tuples (binary), this enforces
    maxsize constraints on the individual arguments of operators.
    E.g., `'pow': (-1, 1)` says that power laws can have any
    complexity left argument, but only 1 complexity in the right
    argument. Use this to force more interpretable solutions.

nested_constraints : dict[str, dict], default=None
    Specifies how many times a combination of operators can be
    nested. For example, `{"sin": {"cos": 0}}, "cos": {"cos": 2}}`
    specifies that `cos` may never appear within a `sin`, but `sin`
    can be nested with itself an unlimited number of times. The
    second term specifies that `cos` can be nested up to 2 times
    within a `cos`, so that `cos(cos(cos(x)))` is allowed
    (as well as any combination of `+` or `-` within it), but
    `cos(cos(cos(cos(x))))` is not allowed. When an operator is not
    specified, it is assumed that it can be nested an unlimited
    number of times. This requires that there is no operator which
    is used both in the unary operators and the binary operators
    (e.g., `-` could be both subtract, and negation). For binary
    operators, you only need to provide a single number: both
    arguments are treated the same way, and the max of each
    argument is constrained.

loss : str, default="L2DistLoss()"
    String of Julia code specifying the loss function. Can either
    be a loss from LossFunctions.jl, or your own loss written as a
    function. Examples of custom written losses include:
    `myloss(x, y) = abs(x-y)` for non-weighted, or
    `myloss(x, y, w) = w*abs(x-y)` for weighted.
    The included losses include:
    Regression: `LPDistLoss{P}()`, `L1DistLoss()`,
    `L2DistLoss()` (mean square), `LogitDistLoss()`,
    `HuberLoss(d)`, `L1EpsilonInsLoss(ϵ)`, `L2EpsilonInsLoss(ϵ)`,
    `PeriodicLoss(c)`, `QuantileLoss(τ)`.
    Classification: `ZeroOneLoss()`, `PerceptronLoss()`,
    `L1HingeLoss()`, `SmoothedL1HingeLoss(γ)`,
    `ModifiedHuberLoss()`, `L2MarginLoss()`, `ExpLoss()`,
    `SigmoidLoss()`, `DWDMarginLoss(q)`.

complexity_of_operators : dict[str, float], default=None
    If you would like to use a complexity other than 1 for an
    operator, specify the complexity here. For example,
    `{"sin": 2, "+": 1}` would give a complexity of 2 for each use
    of the `sin` operator, and a complexity of 1 for each use of
    the `+` operator (which is the default). You may specify real
    numbers for a complexity, and the total complexity of a tree
    will be rounded to the nearest integer after computing.

complexity_of_constants : float, default=1
    Complexity of constants.

complexity_of_variables : float, default=1
    Complexity of variables.

parsimony : float, default=0.0032
    Multiplicative factor for how much to punish complexity.

use_frequency : bool, default=True
    Whether to measure the frequency of complexities, and use that
    instead of parsimony to explore equation space. Will naturally
    find equations of all complexities.

use_frequency_in_tournament : bool, default=True
    Whether to use the frequency mentioned above in the tournament,
    rather than just the simulated annealing.

alpha : float, default=0.1
    Initial temperature for simulated annealing
    (requires :param`annealing` to be `True`).

annealing : bool, default=False
    Whether to use annealing.

early_stop_condition : { float | str }, default=None
    Stop the search early if this loss is reached. You may also
    pass a string containing a Julia function which
    takes a loss and complexity as input, for example:
    `"f(loss, complexity) = (loss < 0.1) && (complexity < 10)"`.

ncyclesperiteration : int, default=550
    Number of total mutations to run, per 10 samples of the
    population, per iteration.

fraction_replaced : float, default=0.000364
    How much of population to replace with migrating equations from
    other populations.

fraction_replaced_hof : float, default=0.035
    How much of population to replace with migrating equations from
    hall of fame.

weight_add_node : float, default=0.79
    Relative likelihood for mutation to add a node.

weight_insert_node : float, default=5.1
    Relative likelihood for mutation to insert a node.

weight_delete_node : float, default=1.7
    Relative likelihood for mutation to delete a node.

weight_do_nothing : float, default=0.21
    Relative likelihood for mutation to leave the individual.

weight_mutate_constant : float, default=0.048
    Relative likelihood for mutation to change the constant slightly
    in a random direction.

weight_mutate_operator : float, default=0.47
    Relative likelihood for mutation to swap an operator.

weight_randomize : float, default=0.00023
    Relative likelihood for mutation to completely delete and then
    randomly generate the equation

weight_simplify : float, default=0.0020
    Relative likelihood for mutation to simplify constant parts by evaluation

crossover_probability : float, default=0.066
    Absolute probability of crossover-type genetic operation, instead of a mutation.

skip_mutation_failures : bool, default=True
    Whether to skip mutation and crossover failures, rather than
    simply re-sampling the current member.

migration : bool, default=True
    Whether to migrate.

hof_migration : bool, default=True
    Whether to have the hall of fame migrate.

topn : int, default=12
    How many top individuals migrate from each population.

should_optimize_constants : bool, default=True
    Whether to numerically optimize constants (Nelder-Mead/Newton)
    at the end of each iteration.

optimizer_algorithm : str, default="BFGS"
    Optimization scheme to use for optimizing constants. Can currently
    be `NelderMead` or `BFGS`.

optimizer_nrestarts : int, default=2
    Number of time to restart the constants optimization process with
    different initial conditions.

optimize_probability : float, default=0.14
    Probability of optimizing the constants during a single iteration of
    the evolutionary algorithm.

optimizer_iterations : int, default=8
    Number of iterations that the constants optimizer can take.

perturbation_factor : float, default=0.076
    Constants are perturbed by a max factor of
    (perturbation_factor*T + 1). Either multiplied by this or
    divided by this.

tournament_selection_n : int, default=10
    Number of expressions to consider in each tournament.

tournament_selection_p : float, default=0.86
    Probability of selecting the best expression in each
    tournament. The probability will decay as p*(1-p)^n for other
    expressions, sorted by loss.

procs : int, default=multiprocessing.cpu_count()
    Number of processes (=number of populations running).

multithreading : bool, default=True
    Use multithreading instead of distributed backend.
    Using procs=0 will turn off both.

cluster_manager : str, default=None
    For distributed computing, this sets the job queue system. Set
    to one of "slurm", "pbs", "lsf", "sge", "qrsh", "scyld", or
    "htc". If set to one of these, PySR will run in distributed
    mode, and use `procs` to figure out how many processes to launch.

batching : bool, default=False
    Whether to compare population members on small batches during
    evolution. Still uses full dataset for comparing against hall
    of fame.

batch_size : int, default=50
    The amount of data to use if doing batching.

fast_cycle : bool, default=False (experimental)
    Batch over population subsamples. This is a slightly different
    algorithm than regularized evolution, but does cycles 15%
    faster. May be algorithmically less efficient.

precision : int, default=32
    What precision to use for the data. By default this is 32
    (float32), but you can select 64 or 16 as well.

random_state : int, Numpy RandomState instance or None, default=None
    Pass an int for reproducible results across multiple function calls.
    See :term:`Glossary <random_state>`.

deterministic : bool, default=False
    Make a PySR search give the same result every run.
    To use this, you must turn off parallelism
    (with :param`procs`=0, :param`multithreading`=False),
    and set :param`random_state` to a fixed seed.

warm_start : bool, default=False
    Tells fit to continue from where the last call to fit finished.
    If false, each call to fit will be fresh, overwriting previous results.

verbosity : int, default=1e9
    What verbosity level to use. 0 means minimal print statements.

update_verbosity : int, default=None
    What verbosity level to use for package updates.
    Will take value of :param`verbosity` if not given.

progress : bool, default=True
    Whether to use a progress bar instead of printing to stdout.

equation_file : str, default=None
    Where to save the files (.csv separated by |).

temp_equation_file : bool, default=False
    Whether to put the hall of fame file in the temp directory.
    Deletion is then controlled with the :param`delete_tempfiles`
    parameter.

tempdir : str, default=None
    directory for the temporary files.

delete_tempfiles : bool, default=True
    Whether to delete the temporary files after finishing.

julia_project : str, default=None
    A Julia environment location containing a Project.toml
    (and potentially the source code for SymbolicRegression.jl).
    Default gives the Python package directory, where a
    Project.toml file should be present from the install.

update: bool, default=True
    Whether to automatically update Julia packages.

output_jax_format : bool, default=False
    Whether to create a 'jax_format' column in the output,
    containing jax-callable functions and the default parameters in
    a jax array.

output_torch_format : bool, default=False
    Whether to create a 'torch_format' column in the output,
    containing a torch module with trainable parameters.

extra_sympy_mappings : dict[str, Callable], default=None
    Provides mappings between custom :param`binary_operators` or
    :param`unary_operators` defined in julia strings, to those same
    operators defined in sympy.
    E.G if `unary_operators=["inv(x)=1/x"]`, then for the fitted
    model to be export to sympy, :param`extra_sympy_mappings`
    would be `{"inv": lambda x: 1/x}`.

extra_jax_mappings : dict[Callable, str], default=None
    Similar to :param`extra_sympy_mappings` but for model export
    to jax. The dictionary maps sympy functions to jax functions.
    For example: `extra_jax_mappings={sympy.sin: "jnp.sin"}` maps
    the `sympy.sin` function to the equivalent jax expression `jnp.sin`.

extra_torch_mappings : dict[Callable, Callable], default=None
    The same as :param`extra_jax_mappings` but for model export
    to pytorch. Note that the dictionary keys should be callable
    pytorch expressions.
    For example: `extra_torch_mappings={sympy.sin: torch.sin}`

denoise : bool, default=False
    Whether to use a Gaussian Process to denoise the data before
    inputting to PySR. Can help PySR fit noisy data.

select_k_features : int, default=None
     whether to run feature selection in Python using random forests,
     before passing to the symbolic regression code. None means no
     feature selection; an int means select that many features.

kwargs : dict, default=None
    Supports deprecated keyword arguments. Other arguments will
    result in an error.

Attributes
----------
equations_ : { pandas.DataFrame | list[pandas.DataFrame] }
    Processed DataFrame containing the results of model fitting.

n_features_in_ : int
    Number of features seen during :term:`fit`.

feature_names_in_ : ndarray of shape (`n_features_in_`,)
    Names of features seen during :term:`fit`. Defined only when `X`
    has feature names that are all strings.

nout_ : int
    Number of output dimensions.

selection_mask_ : list[int] of length `select_k_features`
    List of indices for input features that are selected when
    :param`select_k_features` is set.

tempdir_ : Path
    Path to the temporary equations directory.

equation_file_ : str
    Output equation file name produced by the julia backend.

raw_julia_state_ : tuple[list[PyCall.jlwrap], PyCall.jlwrap]
    The state for the julia SymbolicRegression.jl backend post fitting.

equation_file_contents_ : list[pandas.DataFrame]
    Contents of the equation file output by the Julia backend.

Notes
-----
Most default parameters have been tuned over several example equations,
but you should adjust `niterations`, `binary_operators`, `unary_operators`
to your requirements. You can view more detailed explanations of the options
on the [options page](https://astroautomata.com/PySR/#/options) of the
documentation.

Examples
--------
```python
>>> import numpy as np
>>> from pysr import PySRRegressor
>>> randstate = np.random.RandomState(0)
>>> X = 2 * randstate.randn(100, 5)
>>> # y = 2.5382 * cos(x_3) + x_0 - 0.5
>>> y = 2.5382 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 0.5
>>> model = PySRRegressor(
...     niterations=40,
...     binary_operators=["+", "*"],
...     unary_operators=[
...         "cos",
...         "exp",
...         "sin",
...         "inv(x) = 1/x",  # Custom operator (julia syntax)
...     ],
...     model_selection="best",
...     loss="loss(x, y) = (x - y)^2",  # Custom loss function (julia syntax)
... )
>>> model.fit(X, y)
>>> model
PySRRegressor.equations_ = [
0         0.000000                                          3.8552167  3.360272e+01           1
1         1.189847                                          (x0 * x0)  3.110905e+00           3
2         0.010626                          ((x0 * x0) + -0.25573406)  3.045491e+00           5
3         0.896632                              (cos(x3) + (x0 * x0))  1.242382e+00           6
4         0.811362                ((x0 * x0) + (cos(x3) * 2.4384754))  2.451971e-01           8
5  >>>>  13.733371          (((cos(x3) * 2.5382) + (x0 * x0)) + -0.5)  2.889755e-13          10
6         0.194695  ((x0 * x0) + (((cos(x3) + -0.063180044) * 2.53...  1.957723e-13          12
7         0.006988  ((x0 * x0) + (((cos(x3) + -0.32505524) * 1.538...  1.944089e-13          13
8         0.000955  (((((x0 * x0) + cos(x3)) + -0.8251649) + (cos(...  1.940381e-13          15
]
>>> model.score(X, y)
1.0
>>> model.predict(np.array([1,2,3,4,5]))
array([-1.15907818, -1.15907818, -1.15907818, -1.15907818, -1.15907818])
```


<a id="pysr.sr.PySRRegressor.__repr__"></a>

#### \_\_repr\_\_

```python
def __repr__()
```

[[view_source]](https://github.com/MilesCranmer/PySR/blob/8da5000dfc58c1a035c623ebd6c6e2acea472134/pysr/sr.py#L808)

Prints all current equations fitted by the model.

The string `>>>>` denotes which equation is selected by the
`model_selection`.

<a id="pysr.sr.PySRRegressor.get_best"></a>

#### get\_best

```python
def get_best(index=None)
```

[[view_source]](https://github.com/MilesCranmer/PySR/blob/8da5000dfc58c1a035c623ebd6c6e2acea472134/pysr/sr.py#L918)

Get best equation using `model_selection`.

Parameters
----------
index : int | list[int], default=None
    If you wish to select a particular equation from `self.equations_`,
    give the row number here. This overrides the :param`model_selection`
    parameter. If there are multiple output features, then pass
    a list of indices with the order the same as the output feature.

Returns
-------
best_equation : pandas.Series
    Dictionary representing the best expression found.

Raises
------
NotImplementedError
    Raised when an invalid model selection strategy is provided.


<a id="pysr.sr.PySRRegressor.fit"></a>

#### fit

```python
def fit(X, y, Xresampled=None, weights=None, variable_names=None)
```

[[view_source]](https://github.com/MilesCranmer/PySR/blob/8da5000dfc58c1a035c623ebd6c6e2acea472134/pysr/sr.py#L1496)

Search for equations to fit the dataset and store them in `self.equations_`.

Parameters
----------
X : {ndarray | pandas.DataFrame} of shape (n_samples, n_features)
    Training data.

y : {ndarray | pandas.DataFrame} of shape (n_samples,) or (n_samples, n_targets)
    Target values. Will be cast to X's dtype if necessary.

Xresampled : {ndarray | pandas.DataFrame} of shape
                (n_resampled, n_features), default=None
    Resampled training data to generate a denoised data on. This
    will be used as the training data, rather than `X`.

weights : {ndarray | pandas.DataFrame} of the same shape as y, default=None
    Each element is how to weight the mean-square-error loss
    for that particular element of `y`. Alternatively,
    if a custom `loss` was set, it will can be used
    in arbitrary ways.

variable_names : list[str], default=None
    A list of names for the variables, rather than "x0", "x1", etc.
    If :param`X` is a pandas dataframe, the column names will be used
    instead of `variable_names`. Cannot contain spaces or special
    characters. Avoid variable names which are also
    function names in `sympy`, such as "N".

Returns
-------
self : object
    Fitted estimator.


<a id="pysr.sr.PySRRegressor.refresh"></a>

#### refresh

```python
def refresh(checkpoint_file=None)
```

[[view_source]](https://github.com/MilesCranmer/PySR/blob/8da5000dfc58c1a035c623ebd6c6e2acea472134/pysr/sr.py#L1612)

Updates self.equations_ with any new options passed, such as


<a id="pysr.sr.PySRRegressor.predict"></a>

#### predict

```python
def predict(X, index=None)
```

[[view_source]](https://github.com/MilesCranmer/PySR/blob/8da5000dfc58c1a035c623ebd6c6e2acea472134/pysr/sr.py#L1628)

Predict y from input X using the equation chosen by `model_selection`.

You may see what equation is used by printing this object. X should
have the same columns as the training data.

Parameters
----------
X : {ndarray | pandas.DataFrame} of shape (n_samples, n_features)
    Training data.

index : int | list[int], default=None
    If you want to compute the output of an expression using a
    particular row of `self.equations_`, you may specify the index here.
    For multiple output equations, you must pass a list of indices
    in the same order.

Returns
-------
y_predicted : ndarray of shape (n_samples, nout_)
    Values predicted by substituting `X` into the fitted symbolic
    regression model.

Raises
------
ValueError
    Raises if the `best_equation` cannot be evaluated.

<a id="pysr.sr.PySRRegressor.sympy"></a>

#### sympy

```python
def sympy(index=None)
```

[[view_source]](https://github.com/MilesCranmer/PySR/blob/8da5000dfc58c1a035c623ebd6c6e2acea472134/pysr/sr.py#L1701)

Return sympy representation of the equation(s) chosen by `model_selection`.

Parameters
----------
index : int | list[int], default=None
    If you wish to select a particular equation from
    `self.equations_`, give the index number here. This overrides
    the `model_selection` parameter. If there are multiple output
    features, then pass a list of indices with the order the same
    as the output feature.

Returns
-------
best_equation : str, list[str] of length nout_
    SymPy representation of the best equation.

<a id="pysr.sr.PySRRegressor.latex"></a>

#### latex

```python
def latex(index=None)
```

[[view_source]](https://github.com/MilesCranmer/PySR/blob/8da5000dfc58c1a035c623ebd6c6e2acea472134/pysr/sr.py#L1725)

Return latex representation of the equation(s) chosen by `model_selection`.

Parameters
----------
index : int | list[int], default=None
    If you wish to select a particular equation from
    `self.equations_`, give the index number here. This overrides
    the `model_selection` parameter. If there are multiple output
    features, then pass a list of indices with the order the same
    as the output feature.

Returns
-------
best_equation : str or list[str] of length nout_
    LaTeX expression of the best equation.

<a id="pysr.sr.PySRRegressor.jax"></a>

#### jax

```python
def jax(index=None)
```

[[view_source]](https://github.com/MilesCranmer/PySR/blob/8da5000dfc58c1a035c623ebd6c6e2acea472134/pysr/sr.py#L1749)

Return jax representation of the equation(s) chosen by `model_selection`.

Each equation (multiple given if there are multiple outputs) is a dictionary
containing {"callable": func, "parameters": params}. To call `func`, pass
func(X, params). This function is differentiable using `jax.grad`.

Parameters
----------
index : int | list[int], default=None
    If you wish to select a particular equation from
    `self.equations_`, give the index number here. This overrides
    the `model_selection` parameter. If there are multiple output
    features, then pass a list of indices with the order the same
    as the output feature.

Returns
-------
best_equation : dict[str, Any]
    Dictionary of callable jax function in "callable" key,
    and jax array of parameters as "parameters" key.

<a id="pysr.sr.PySRRegressor.pytorch"></a>

#### pytorch

```python
def pytorch(index=None)
```

[[view_source]](https://github.com/MilesCranmer/PySR/blob/8da5000dfc58c1a035c623ebd6c6e2acea472134/pysr/sr.py#L1779)

Return pytorch representation of the equation(s) chosen by `model_selection`.

Each equation (multiple given if there are multiple outputs) is a PyTorch module
containing the parameters as trainable attributes. You can use the module like
any other PyTorch module: `module(X)`, where `X` is a tensor with the same
column ordering as trained with.

Parameters
----------
index : int | list[int], default=None
    If you wish to select a particular equation from
    `self.equations_`, give the index number here. This overrides
    the `model_selection` parameter. If there are multiple output
    features, then pass a list of indices with the order the same
    as the output feature.

Returns
-------
best_equation : torch.nn.Module
    PyTorch module representing the expression.

