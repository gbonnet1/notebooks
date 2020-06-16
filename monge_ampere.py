# %% [markdown]
"""
# Monge-Ampère equations
"""


# %%
import agd.LinearParallel as lp
import matplotlib.pyplot as plt
import numpy as np
from agd import Domain, Selling
from agd.AutomaticDifferentiation.Optimization import newton_root

# %% [markdown]
"""
In this notebook, we aim to solve Monge-Ampère equations of the form

$$ \det (D^2 u(x) - A(x, u(x), D u(x))) = B(x, u(x), D u(x)) $$

on an open domain $\Omega \subset \mathbb{R}^2$, for some given functions $A \colon \mathbb{R}^2 \times \mathbb{R} \times \mathbb{R}^2 \to \mathcal{S}_2$ and $B \colon \mathbb{R}^2 \times \mathbb{R} \times \mathbb{R}^2 \to \mathbb{R}$. We reformulate the Monge-Ampère equation in the form

$$ \sup_{\substack{\mathcal{D} \in \mathcal{S}_2^+ \\ \operatorname{Tr}(\mathcal{D}) = 1}} 2 (\det \mathcal{D})^{1/2} B(x, u(x), D^2 u(x))^{1/2} - \langle \mathcal{D}, D^2 u(x) - A(x, u(x), D^2 u(x)) \rangle = 0. $$

This reformulation satisfies two properties:

* It is *degenerate elliptic*, meaning that it may be written as

  $$ F(x, u(x), D u(x), D^2 u(x)) = 0, $$

  where the function $F \colon \mathbb{R}^2 \times \mathbb{R} \times \mathbb{R}^2 \times \mathcal{S}_2 \to \mathbb{R}$ is nonincreasing with respect to its last variable.

* It selects *admissible* solutions, that is, solutions $u \colon \Omega \to \mathbb{R}$ such that

  $$ D^2 u(x) - A(x, u(x), D u(x)) \geq 0. $$
"""


# %% [markdown]
"""
## 1. The discretized equation
"""


# %% [markdown]
"""
We discretize the reformulated equation on a grid $\mathcal{G}_h := \Omega \cap h \mathbb{Z}^2$, for some discretization step $h > 0$.
"""


# %%
x = np.stack(np.meshgrid(*(2 * [np.linspace(-1, 1, 100)]), indexing="ij"))


# %% [markdown]
"""
The discretization of the equation is called *monotone* if the resulting numerical scheme may be written as

$$ F^h(x, u^h(x), u^h) = 0, $$

where $F^h \colon \mathcal{G}_h \times \mathbb{R} \times \mathbb{R}^{\mathcal{G}_h} \to \mathbb{R}$ is a function that is nonincreasing with respect to its last argument. Monotonicity is a discrete counterpart to degenerate ellipticity and is sometimes required for the scheme to be convergent.

Let $u \colon \Omega \to \mathbb{R}$ and $x \in \Omega$. We assume for now that $x$ is far from $\partial \Omega$. For any $e \in \mathbb{Z}^d$ such that $B_2(x, h |e|) \in \Omega$, we define

\begin{align*}
\delta_h^e u(x) &:= \frac{u(x + h e) - u(x)}{h}, &
\Delta_h^e u(x) &:= \frac{u(x + h e) + u(x - h e) - 2 u(x)}{h^2}.
\end{align*}

if $u \in C^4(\Omega)$, then

\begin{align*}
\delta_h^e u(x) &= \langle e, D u(x) \rangle + O(h), &
\Delta_h^e u(x) &= \langle e, D^2 u(x) e \rangle + O(h^2).
\end{align*}

We build a monotone finite difference scheme using the notion of superbase of $\mathbb{Z}^2$. A basis of $\mathbb{Z}^2$ is a pair of $v = (v_1, v_2)$ of vectors of $\mathbb{Z}^2$ such that $\det(v_1, v_2) = \pm 1$. A superbase of $\mathbb{Z}^2$ is a triplet $v = (v_1, v_2, v_3)$ of vectors of $\mathbb{Z}^2$ such that $(v_1, v_2)$ is a basis of $\mathbb{Z}^2$ and $v_3 = -v_1 - v_2$. Note that the definition is symmetric: $(v_1, v_3)$ and $(v_2, v_3)$ are also bases of $\mathbb{Z}^2$.

Let $\mathcal{D} \in \mathcal{S}_2$. If $v$ is a superbase of $\mathbb{Z}^2$, then we have Selling's formula:

$$ \mathcal{D} = -\sum_{1 \leq i \leq 3} \langle v_{i+1}, \mathcal{D} v_{i+2} \rangle v_i^\perp (v_i^\perp)^\top, $$

where indices are taken modulo three. The superbase $v$ is called *$\mathcal{D}$-obtuse* if $\langle v_i, \mathcal{D} v_j \rangle \leq 0$ for any $1 \leq i < j \leq 3$. If $v$ is $\mathcal{D}$-obtuse, we define the finite difference operator

$$ \Delta_h^{\mathcal{D}} u(x) := -\sum_{1 \leq i \leq 3} \langle v_{i+1}, \mathcal{D} v_{i+2} \rangle \Delta_h^{v_i^\perp} u(x). $$

By Selling's formula, if $u \in C^4(\Omega)$, then

$$ \Delta_h^{\mathcal{D}} u(x) = \langle \mathcal{D}, D^2 u(x) \rangle + O(h^2). $$

The $\mathcal{D}$-obtuseness of $v$ is required so that the operator $\Delta_h^{\mathcal{D}}$ may be used to build a monotone scheme. For any superbase $v$ of $\mathbb{Z}^2$, we define the set $\mathcal{S}_2^v \subset \mathcal{S}_d^+$ of matrices $\mathcal{D} \in \mathcal{S}_d^+$ such that $v$ is $\mathcal{D}$-obtuse. For any $\mu \geq 1$, Selling's algorithm maybe used to compute a set $V$ of superbases of $\mathbb{Z}^2$ such that $\cup_{v \in V} \mathcal{S}_2^v$ contains all matrices $\mathcal{D} \in \mathcal{S}_2^{++}$ whose condition number is less than or equal to $\mu$.
"""


# %%
superbases = np.multiply.outer(
    Selling.SuperbasesForConditioning(15), np.ones(x.shape[1:], dtype=np.int64)
)


# %%
def MA(A, B, d2u, superbases):
    delta = d2u - lp.dot_VAV(
        lp.perp(superbases), A[:, :, np.newaxis, np.newaxis], lp.perp(superbases)
    )

    residue = -np.inf

    W = (
        -np.stack(
            [
                np.roll(superbases[0], 1, axis=0) * np.roll(superbases[0], 2, axis=0)
                - np.roll(superbases[1], 1, axis=0) * np.roll(superbases[1], 2, axis=0),
                np.roll(superbases[0], 1, axis=0) * np.roll(superbases[1], 2, axis=0)
                + np.roll(superbases[1], 1, axis=0) * np.roll(superbases[0], 2, axis=0),
            ]
        )
        / 2
    )
    w = -lp.dot_VV(np.roll(superbases, 1, axis=1), np.roll(superbases, 2, axis=1)) / 2

    q = lp.dot_AV(W, delta)
    r = np.sqrt(B + lp.dot_VV(q, q))

    residue = np.maximum(
        residue,
        np.max(
            np.where(
                np.all(lp.dot_VA(q, W) <= r * w, axis=0),
                r - lp.dot_VV(w, delta),
                -np.inf,
            ),
            axis=0,
        ),
    )

    bases = np.concatenate(
        [superbases[:, [0, 1]], superbases[:, [0, 2]], superbases[:, [1, 2]]], axis=2
    )
    delta_bases = np.concatenate([delta[[0, 1]], delta[[0, 2]], delta[[1, 2]]], axis=1)

    residue = np.maximum(
        residue,
        np.max(
            np.sqrt(
                B
                * lp.det(bases) ** 2
                / (
                    lp.dot_VV(bases[:, 0], bases[:, 0])
                    * lp.dot_VV(bases[:, 1], bases[:, 1])
                )
                + (
                    delta_bases[0] / lp.dot_VV(bases[:, 0], bases[:, 0])
                    - delta_bases[1] / lp.dot_VV(bases[:, 1], bases[:, 1])
                )
                ** 2
                / 4
            )
            - (
                delta_bases[0] / lp.dot_VV(bases[:, 0], bases[:, 0])
                + delta_bases[1] / lp.dot_VV(bases[:, 1], bases[:, 1])
            )
            / 2,
            axis=0,
        ),
    )

    return residue


# %% [markdown]
"""
## 2. Dirichlet boundary conditions
"""


# %%
def SchemeDirichlet(u, x, domain, A, B, g, superbases):
    bc = Domain.Dirichlet(domain, g, x)

    du = bc.DiffCentered(u, [[1, 0], [0, 1]])
    d2u = bc.Diff2(u, lp.perp(superbases))

    return np.where(
        bc.interior, MA(A(x, u, du), B(x, u, du), d2u, superbases), u - bc.grid_values
    )


# %%
domain = Domain.Box([[-1, 1], [-1, 1]])


def A(x, r, p):
    return np.zeros((2, 2) + x.shape[1:])


def B(x, r, p):
    return np.ones(x.shape[1:])


u = newton_root(
    SchemeDirichlet, np.zeros(x.shape[1:]), (x, domain, A, B, 0.0, superbases)
)
u = np.where(domain.level(x) < 0, u, np.nan)

plt.contourf(*x, u)
plt.show()


# %% [markdown]
"""
### 2.1. Comparison with the exact solution
"""


# %%
domain = Domain.Ball()


def A(x, r, p):
    return np.zeros((2, 2) + x.shape[1:])


def B(x, r, p):
    return (4 + 32 * lp.dot_VV(x, x) + 48 * lp.dot_VV(x, x) ** 2) / 36


def Exact(x):
    return (lp.dot_VV(x, x) + lp.dot_VV(x, x) ** 2) / 6


u = newton_root(
    SchemeDirichlet, np.zeros(x.shape[1:]), (x, domain, A, B, 1 / 3, superbases)
)
u = np.where(domain.level(x) < 0, u, np.nan)

plt.contourf(*x, u)
plt.show()

err = np.where(domain.level(x) < 0, u - Exact(x), 0)
print("Error:", np.max(np.abs(err)))


# %%
Q = np.array([[2, 1], [1, 2]])
Q_inv = np.array([[2, -1], [-1, 2]]) / 3

assert np.allclose(Q @ Q_inv, np.eye(2))

domain = Domain.Ball()


def A(x, r, p):
    return -(
        r / 3 + lp.dot_VAV(p, np.multiply.outer(Q_inv, np.ones(x.shape[1:])), p) / 3
    ) * np.multiply.outer(Q, np.ones(x.shape[1:]))


def B(x, r, p):
    return (
        3
        * (
            1
            + 2 * r / 3
            + lp.dot_VAV(p, np.multiply.outer(Q_inv, np.ones(x.shape[1:])), p) / 6
        )
        ** 2
    )


def Exact(x):
    return lp.dot_VAV(x, np.multiply.outer(Q, np.ones(x.shape[1:])), x) / 2


u = newton_root(
    SchemeDirichlet, np.zeros(x.shape[1:]), (x, domain, A, B, Exact, superbases)
)
u = np.where(domain.level(x) < 0, u, np.nan)

plt.contourf(*x, u)
plt.show()

err = np.where(domain.level(x) < 0, u - Exact(x), 0)
print("Error:", np.max(np.abs(err)))


# %% [markdown]
"""
### 2.2. Other domains
"""


# %%
domain = Domain.Union(Domain.Ball(), Domain.Box())


def A(x, r, p):
    return np.zeros((2, 2) + x.shape[1:])


def B(x, r, p):
    return np.ones(x.shape[1:])


u = newton_root(
    SchemeDirichlet, np.zeros(x.shape[1:]), (x, domain, A, B, 0.0, superbases)
)
u = np.where(domain.level(x) < 0, u, np.nan)

plt.contourf(*x, u)
plt.show()


# %%
domain = Domain.Complement(Domain.Ball(), Domain.Box())


def A(x, r, p):
    return np.zeros((2, 2) + x.shape[1:])


def B(x, r, p):
    return np.ones(x.shape[1:])


u = newton_root(
    SchemeDirichlet, np.zeros(x.shape[1:]), (x, domain, A, B, 0.0, superbases)
)
u = np.where(domain.level(x) < 0, u, np.nan)

plt.contourf(*x, u)
plt.show()


# %% [markdown]
"""
## 3. Optimal transport boundary conditions
"""


# %%
def SchemeBV2(u, x, domain, A, B, C, sigma, superbases):
    bc = Domain.Dirichlet(domain, np.inf, x)

    du0 = bc.DiffUpwind(u, [[1, 0], [0, 1]])
    du1 = bc.DiffUpwind(u, [[-1, 0], [0, -1]])
    # TODO
    assert np.sum(np.logical_and(du0 == np.inf, du1 == np.inf)) == 0
    du0 = np.where(du0 == np.inf, -du1, du0)
    du1 = np.where(du1 == np.inf, -du0, du1)
    du = (du0 - du1) / 2

    du0 = bc.DiffUpwind(u, lp.perp(superbases))
    du0 = np.where(
        du0 == np.inf,
        sigma(
            x[:, np.newaxis, np.newaxis],
            u[np.newaxis, np.newaxis],
            lp.perp(superbases),
        ),
        du0,
    )
    du1 = bc.DiffUpwind(u, -lp.perp(superbases))
    du1 = np.where(
        du1 == np.inf,
        sigma(
            x[:, np.newaxis, np.newaxis],
            u[np.newaxis, np.newaxis],
            -lp.perp(superbases),
        ),
        du1,
    )
    d2u = (du0 + du1) / bc.gridscale

    return np.where(
        bc.interior,
        MA(A(x, u, du), B(x, u, du), d2u, superbases)
        + u.flatten()[np.argmin(bc.domain.level(bc.grid))]
        - C,
        u - bc.grid_values,
    )


# %% [markdown]
"""
### 3.1. Comparison with the exact solution
"""


# %%
domain = Domain.Ball()


def A(x, r, p):
    return np.zeros((2, 2) + x.shape[1:])


def B(x, r, p):
    return (4 + 32 * lp.dot_VV(x, x) + 48 * lp.dot_VV(x, x) ** 2) / 36


def sigma(x, r, e):
    return np.sqrt(lp.dot_VV(e, e))


def Exact(x):
    return (lp.dot_VV(x, x) + lp.dot_VV(x, x) ** 2) / 6


u = newton_root(
    SchemeBV2, np.zeros(x.shape[1:]), (x, domain, A, B, 0, sigma, superbases)
)
u = np.where(domain.level(x) < 0, u, np.nan)

plt.contourf(*x, u)
plt.show()

err = np.where(domain.level(x) < 0, u - Exact(x), 0)
print("Error:", np.max(np.abs(err)))


# %%
domain = Domain.Ball()


def A(x, r, p):
    return -(r / 3 + lp.dot_VV(p, p) / 3) * lp.identity(x.shape[1:])


def B(x, r, p):
    return (1 + 2 * r / 3 + lp.dot_VV(p, p) / 6) ** 2


def sigma(x, r, e):
    return np.sqrt(lp.dot_VV(e, e))


def Exact(x):
    return lp.dot_VV(x, x) / 2


# TODO: initial guess
u = newton_root(SchemeBV2, lp.dot_VV(x, x), (x, domain, A, B, 0, sigma, superbases))
u = np.where(domain.level(x) < 0, u, np.nan)

plt.contourf(*x, u)
plt.show()

err = np.where(domain.level(x) < 0, u - Exact(x), 0)
print("Error:", np.max(np.abs(err)))


# %% [markdown]
"""
### 3.2. Near-field reflector design
"""


# %%
domain = Domain.Ball()


def f(x):
    return np.ones(x.shape[1:])


def A(x, r, p):
    tmp = 1 + np.sqrt(1 - lp.dot_VV(p, p) / r ** 4)
    return (2 + tmp) / r * lp.outer(p, p) - r ** 3 * tmp * lp.identity(x.shape[1:])


def B(x, r, p):
    tmp = 1 + np.sqrt(1 - lp.dot_VV(p, p) / r ** 4)
    return r ** 6 * (tmp ** 3 - tmp ** 2) * f(x)


def sigma(x, r, e):
    return 2 * r ** 3 * (np.sqrt(lp.dot_VV(e, e)) - lp.dot_VV(x, e))


u = newton_root(
    SchemeBV2, np.full(x.shape[1:], 0.1), (x, domain, A, B, 0.1, sigma, superbases),
)
u = np.where(domain.level(x) < 0, u, np.nan)

plt.contourf(*x, u)
plt.show()


# %%
