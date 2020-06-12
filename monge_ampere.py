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
def MA(u, A, B, bc, superbases):
    superbases = np.multiply.outer(superbases, np.ones(u.shape, dtype=np.int64))

    du = bc.DiffCentered(u, [[1, 0], [0, 1]])
    d2u = bc.Diff2(u, lp.perp(superbases))

    a = A(bc.grid, u, du)
    b = B(bc.grid, u, du)

    delta = d2u - lp.dot_VAV(
        lp.perp(superbases), a[:, :, np.newaxis, np.newaxis], lp.perp(superbases)
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
    r = np.sqrt(b + lp.dot_VV(q, q))

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
                b
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
def SchemeDirichlet(u, A, B, bc, superbases):
    return np.where(bc.interior, MA(u, A, B, bc, superbases), u - bc.grid_values)


# %%
def A(x, r, p):
    return np.zeros((2, 2) + x.shape[1:])


def B(x, r, p):
    return np.ones(x.shape[1:])


bc = Domain.Dirichlet(Domain.Box([[-1, 1], [-1, 1]]), 0.0, x)

u = newton_root(SchemeDirichlet, np.zeros(x.shape[1:]), (A, B, bc, superbases))

plt.contourf(*x, u)
plt.show()


# %% [markdown]
"""
### 2.1. Comparison with the exact solution
"""


# %%
def Exact(x):
    return (lp.dot_VV(x, x) + lp.dot_VV(x, x) ** 2) / 6


def A(x, r, p):
    return np.zeros((2, 2) + x.shape[1:])


def B(x, r, p):
    return (4 + 32 * lp.dot_VV(x, x) + 48 * lp.dot_VV(x, x) ** 2) / 36


bc = Domain.Dirichlet(Domain.Ball(), 1 / 3, x)

u = newton_root(SchemeDirichlet, np.zeros(x.shape[1:]), (A, B, bc, superbases))

plt.contourf(*x, u)
plt.show()

err = np.where(bc.interior, u - Exact(x), 0)
print("Error:", np.max(np.abs(err)))


# %%
Q = np.array([[2, 1], [1, 2]])
Q_inv = np.array([[2, -1], [-1, 2]]) / 3

assert np.allclose(Q @ Q_inv, np.eye(2))


def Exact(x):
    return lp.dot_VAV(x, np.multiply.outer(Q, np.ones(x.shape[1:])), x) / 2


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


bc = Domain.Dirichlet(Domain.Ball(), Exact, x)

u = newton_root(SchemeDirichlet, np.zeros(x.shape[1:]), (A, B, bc, superbases))

plt.contourf(*x, u)
plt.show()

err = np.where(bc.interior, u - Exact(x), 0)
print("Error:", np.max(np.abs(err)))


# %% [markdown]
"""
### 2.2. Other domains
"""


# %%
def A(x, r, p):
    return np.zeros((2, 2) + x.shape[1:])


def B(x, r, p):
    return np.ones(x.shape[1:])


bc = Domain.Dirichlet(Domain.Union(Domain.Ball(), Domain.Box()), 0.0, x)

u = newton_root(SchemeDirichlet, np.zeros(x.shape[1:]), (A, B, bc, superbases))

plt.contourf(*x, u)
plt.show()


# %%
def A(x, r, p):
    return np.zeros((2, 2) + x.shape[1:])


def B(x, r, p):
    return np.ones(x.shape[1:])


bc = Domain.Dirichlet(Domain.Complement(Domain.Ball(), Domain.Box()), 0.0, x)

u = newton_root(SchemeDirichlet, np.zeros(x.shape[1:]), (A, B, bc, superbases))

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

    def DiffCentered(self, u, offsets):
        du0 = super().DiffUpwind(u, offsets)
        du1 = super().DiffUpwind(u, -np.asarray(offsets))

        # TODO
        assert np.sum(np.logical_and(du0 == np.inf, du1 == np.inf)) == 0

        du0 = np.where(du0 == np.inf, -du1, du0)
        du1 = np.where(du1 == np.inf, -du0, du1)

        return (du0 - du1) / 2


# %%
def SchemeBV2(u, A, B, bc, superbases):
    return np.where(
        bc.interior,
        MA(u, A, B, bc, superbases) + u.flatten()[np.argmin(bc.domain.level(bc.grid))],
        u - bc.grid_values,
    )


# %% [markdown]
"""
### 3.1. Comparison with the exact solution
"""


# %%
def Exact(x):
    return (lp.dot_VV(x, x) + lp.dot_VV(x, x) ** 2) / 6


def A(x, r, p):
    return np.zeros((2, 2) + x.shape[1:])


def B(x, r, p):
    return (4 + 32 * lp.dot_VV(x, x) + 48 * lp.dot_VV(x, x) ** 2) / 36


bc = BV2(Domain.Ball(), x)

u = newton_root(SchemeBV2, np.zeros(x.shape[1:]), (A, B, bc, superbases))

plt.contourf(*x, u)
plt.show()

err = np.where(bc.interior, u - Exact(x), 0)
print("Error:", np.max(np.abs(err)))


# %%
def Exact(x):
    return lp.dot_VV(x, x) / 2


def A(x, r, p):
    return -(r / 3 + lp.dot_VV(p, p) / 3) * lp.identity(x.shape[1:])


def B(x, r, p):
    return (1 + 2 * r / 3 + lp.dot_VV(p, p) / 6) ** 2


bc = BV2(Domain.Ball(), x)

# TODO: initial guess
u = newton_root(SchemeBV2, lp.dot_VV(x, x), (A, B, bc, superbases))

plt.contourf(*x, u)
plt.show()

err = np.where(bc.interior, u - Exact(x), 0)
print("Error:", np.max(np.abs(err)))


# %%
