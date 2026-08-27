# bsde-tt-jax

A JAX implementation of a **Tensor-Train Picard scheme** for solving decoupled forward-backward SDEs (BSDEs)

$$
Y_t = \xi + \int_t^T f(s, Y_s, Z_s)\,ds - \int_t^T Z_s\, dW_s ,
$$

with vector-valued terminal condition $\xi \in \mathbb R^d$ and driver $f(t,y,z): [0,T]\times\mathbb R^d\times\mathbb R^{d\times d_b}\to\mathbb R^d$, driven by a $d_b$-dimensional Brownian motion $W$.

Problems are specified declaratively in a YAML file (one file can contain several BSDEs), solved via Picard iteration where each step's chaos expansion is fit in **Tensor-Train (TT) format using ALS with Christoffel-optimal sampling**, and results/diagnostics are written out per BSDE.

---

## 1. The method

### 1.1 Discretization and the Picard recursion

Fix a time grid $t_0=0 < t_1 < \dots < t_N = T$ with step $h = T/N$, and standardized Brownian increments $\xi_{i,l} = \Delta W_{i,l}/\sqrt h \sim \mathcal N(0,1)$ i.i.d., for time step $i=1,\dots,N$ and Brownian component $l=1,\dots,d_b$.

The scheme is Picard iteration in function space, discretized in time:

$$
Y^{(0)} = 0 \in \mathbb R^d, \qquad Z^{(0)} = 0 \in \mathbb R^{d\times d_b}
$$

For $q = 0,\dots,K-1$:

1. Form the $\mathbb R^d$-valued random variable
$$
F^{(q)} = \xi + \sum_{i=1}^N h\, f\!\big(t_i, Y_{t_i}^{(q)}, Z_{t_i}^{(q)}\big),
$$
a function of the $N\cdot d_b$ i.i.d. Gaussian factors $(\xi_{i,l})$.

2. Compute its **Wiener-chaos (Hermite polynomial) expansion in Tensor-Train format**, $\tilde F^{(q)}$, by ALS with Christoffel sampling

3. Update, for $j=1,\dots,N$:
$$
Y_{t_j}^{(q+1)} = \mathbb E\big[\tilde F^{(q)} \mid \mathcal F_{t_j}\big] - \sum_{i=1}^j h\, f\!\big(t_i, Y_{t_i}^{(q)}, Z_{t_i}^{(q)}\big), \qquad Z_{t_j}^{(q+1)} = D_{t_j}\, \mathbb E\big[\tilde F^{(q)} \mid \mathcal F_{t_j}\big],
$$
where $D_{t_j}$ is the Malliavin derivative.

4. Read off the initial values directly from the PCE coefficients of $\tilde F^{(q)}$:
$$
Y_{t_0}^{(q+1)} = \tilde f^{(q)}_{0,\dots,0}, \qquad Z_{t_0}^{(q+1)} = \frac{1}{\sqrt h}\Big(\tilde f^{(q)}_{e_{1,l}}\Big)_{l=1}^{d_b},
$$
i.e. the constant coefficient and the first-order coefficients along the first time step's Hermite modes.

Iteration stops when a maximum iteration count is reached.

### 1.3 Fitting the TT via ALS with Christoffel sampling

The TT cores are fit using **Alternating Least Squares (ALS)**: sweep left-to-right and back, at each step freezing every core except one and solving a small weighted least-squares problem for that core, then left/right-orthonormalizing it before moving to the next.

Naively sampling the design points i.i.d. from the base Gaussian measure gives an ill-conditioned local regression. Instead each core's micro-problem is solved with **weighted least squares with Christoffel sampling**. The points are drawn from the *Christoffel sampling measure* of the current local basis, then weighted to debias the estimator. Because the local design factorizes across the current core's position,

$$
d\nu(x) \;\propto\; \underbrace{K_L(x_{<k})}_{\text{left environment}} \cdot \underbrace{K_k(x_k)}_{\text{this core's basis}} \cdot \underbrace{K_R(x_{>k})}_{\text{right environment}} \; d\mu(x),
$$

the mode being optimized is sampled directly (1D), while the correlated left/right blocks are sampled using **sequential conditional sampling. A number of samples in $\mathcal O(m\log(m\eta^{-1}))$ ensures a stable least-squares problem with high-probability, and we use resampling on failure ensures the local Gram matrix is well-conditioned before the regression is solved.

---

## 2. Architecture

```
bsde_tt_jax/
├── basis/
│   └── hermite.py         # normalized Hermite basis functions
├── sampling/
│   └── christoffel.py     # Christoffel-optimal sampler:
├── tt/
│   ├── core.py             # TensorTrain constructors
│   ├── ops.py               # left/right QR orthonormalization, tt_norm_diff, tt_at (point evaluation)
│   ├── als.py               # ALS sweep: micro-solve per core + Christoffel sampler + orthonormalization
│   └── compute.py           # conditional_expectation, malliavin_derivative, evaluate_*, compute_t0
├── solver/
│   ├── problem.py           # BSDEProblem dataclass (xi, f, T, N, d, d_b, chaos/als/christoffel/picard configs)
│   └── picard.py            # fit_tt_chaos (chaos-decomposition step), picard_step, run (outer Picard loop)
├── config/
│   └── loader.py              # YAML -> list[BSDEProblem], compiles xi/f expressions
├── expr/
│   └── compile.py             # safe sympy-based compilation of YAML `xi`/`driver` strings to JAX callables
├── io/
│   └── results.py               # SolverResult container, save/load
main.py # Solves a BSDE
```


---

## 3. YAML schema and example

```yaml
bsde:
  - name: linear-polynomial
    d: 1
    db: 1
    T: 1
    N: 10
    constants:
      alpha: 1.0
      beta: 1.0
      n: 2
    xi: "x**n" # terminal condition, function of W{l}_T
    f: "alpha*y + beta*z[0]" # driver f(t, y, z); y[i], z[i,j], t 
    chaos:
      degree: 2 # per-factor Hermite truncation degree
      max_tt_rank: 3
    als:
      max_iters: 30
      tol: 1e-6
      regularization: 1e-8
    picard:
      max_iters: 6
      tol: 1e-4
    christoffel:
      stability: 0.5 # conditioning target for the local Gram matrix
      prob: 0.5 # failure probability bound for the sample-count estimate
      max_resample: 5
    reference:              # optional -- used only for validation/reporting
      Y0: [...]
      Z0: [[...]]
```

**Notes on expressions:**
- `xi` is a function of the terminal Brownian values `x[i]` ($i=0,\dots,d_b$).
- `driver` is a function of `t`, `y[i]` ($i=0,\dots,d-1$), and `z[i,j]` ($i=0,\dots,d-1$, $j=0,\dots,d_b-1$).
- Any name under `constants:` is available inside both expressions.
- Allowed functions: `maximum`, `minimum`, `exp`, `log`, `sqrt`, `sin`, `cos`, `tanh`, `abs`. Expressions are parsed with `sympy` against a whitelist of symbols — no arbitrary code execution.

---

## 4. Usage

### Command line

```bash
python main.py config.yaml linear-polynomial [--x64] [--output <folder>] [--seed 0]
```

`--x64` enables `float64` in JAX, recommended for numerical stability of the ALS regression (JAX defaults to `float32`).

### Python API

```python
from jax import random
from bsde_tt_jax.config.loader import load_config
from bsde_tt_jax.solver.picard import run

problems = load_config("configs/example.yaml")
problem = problems[0]

key = random.PRNGKey(0)
result = run(key, problem)
```