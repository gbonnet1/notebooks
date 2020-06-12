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

# %%
x = np.stack(np.meshgrid(*(2 * [np.linspace(-1, 1, 100)]), indexing="ij"))

superbases = Selling.SuperbasesForConditioning(15)


# %% [markdown]
"""
## 1. The discretized equation
"""


# %%
def MA(u, B, bc, superbases):
    superbases = np.multiply.outer(superbases, np.ones(u.shape, dtype=np.int64))

    b = B(bc.grid, u)

    residue = -np.inf

    d2u = bc.Diff2(u, lp.perp(superbases))

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

    q = lp.dot_AV(W, d2u)
    r = np.sqrt(b + lp.dot_VV(q, q))

    residue = np.maximum(
        residue,
        np.max(
            np.where(
                np.all(lp.dot_VA(q, W) <= r * w, axis=0),
                r - lp.dot_VV(w, d2u),
                -np.inf,
            ),
            axis=0,
        ),
    )

    bases = np.concatenate(
        [superbases[:, [0, 1]], superbases[:, [0, 2]], superbases[:, [1, 2]]], axis=2
    )
    d2u_bases = np.concatenate([d2u[[0, 1]], d2u[[0, 2]], d2u[[1, 2]]], axis=1)

    residue = np.maximum(
        residue,
        np.max(
            np.sqrt(
                b
                * lp.det(bases) ** 2
                / (
                    lp.dot_VV(bases[:, 0], bases[:, 0])
                    * lp.dot_VV(bases[:, 1], bases[:, 1])
                )
                + (
                    d2u_bases[0] / lp.dot_VV(bases[:, 0], bases[:, 0])
                    - d2u_bases[1] / lp.dot_VV(bases[:, 1], bases[:, 1])
                )
                ** 2
                / 4
            )
            - (
                d2u_bases[0] / lp.dot_VV(bases[:, 0], bases[:, 0])
                + d2u_bases[1] / lp.dot_VV(bases[:, 1], bases[:, 1])
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
def SchemeDirichlet(u, B, bc, superbases):
    return np.where(bc.interior, MA(u, B, bc, superbases), u - bc.grid_values)


# %%
def B(x, r):
    return np.ones(x.shape[1:])


bc = Domain.Dirichlet(Domain.Box([[-1, 1], [-1, 1]]), 0.0, x)

u = newton_root(SchemeDirichlet, np.zeros(x.shape[1:]), (B, bc, superbases))

plt.contourf(*x, u)
plt.show()


# %% [markdown]
"""
### 2.1. Comparison with the exact solution
"""


# %%
def B(x, r):
    return 4 + 32 * lp.dot_VV(x, x) + 48 * lp.dot_VV(x, x) ** 2


bc = Domain.Dirichlet(Domain.Ball(), 2.0, x)

u = newton_root(SchemeDirichlet, np.zeros(x.shape[1:]), (B, bc, superbases))

plt.contourf(*x, u)
plt.show()


def ExactQuartic(x):
    return lp.dot_VV(x, x) + lp.dot_VV(x, x) ** 2


err = np.where(bc.interior, u - ExactQuartic(x), 0)
print("Error:", np.max(np.abs(err)))


# %% [markdown]
"""
### 2.2. Other domains
"""


# %%
def B(x, r):
    return np.ones(x.shape[1:])


bc = Domain.Dirichlet(Domain.Union(Domain.Ball(), Domain.Box()), 0.0, x)

u = newton_root(SchemeDirichlet, np.zeros(x.shape[1:]), (B, bc, superbases))

plt.contourf(*x, u)
plt.show()


# %%
def B(x, r):
    return np.ones(x.shape[1:])


bc = Domain.Dirichlet(Domain.Complement(Domain.Ball(), Domain.Box()), 0.0, x)

u = newton_root(SchemeDirichlet, np.zeros(x.shape[1:]), (B, bc, superbases))

plt.contourf(*x, u)
plt.show()


# %% [markdown]
"""
## 3. Optimal transport boundary conditions
"""


# %%
class BV2(Domain.Dirichlet):
    def __init__(self, domain, grid):
        super().__init__(domain, np.inf, grid)

    def DiffUpwind(self, u, offsets, reth=False):
        du = super().DiffUpwind(u, offsets)

        du = np.where(du == np.inf, np.sqrt(lp.dot_VV(offsets, offsets)), du)

        if reth:
            return du, self.gridscale
        else:
            return du


# %%
def SchemeBV2(u, B, bc, superbases):
    return np.where(
        bc.interior,
        MA(u, B, bc, superbases) + u.flatten()[np.argmin(bc.domain.level(bc.grid))],
        u - bc.grid_values,
    )


# %% [markdown]
"""
### 3.1. Comparison with the exact solution
"""


# %%
def B(x, r):
    return (4 + 32 * lp.dot_VV(x, x) + 48 * lp.dot_VV(x, x) ** 2) / 36


bc = BV2(Domain.Ball(), x)

u = newton_root(SchemeBV2, np.zeros(x.shape[1:]), (B, bc, superbases))

plt.contourf(*x, u)
plt.show()


def ExactQuartic(x):
    return (lp.dot_VV(x, x) + lp.dot_VV(x, x) ** 2) / 6


err = np.where(bc.interior, u - ExactQuartic(x), 0)
print("Error:", np.max(np.abs(err)))


# %%
