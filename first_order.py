# %% [markdown]
"""
# Equations with first order term
"""


# %%
# !test -d /var/colab && pip install poetry
# !test -d /var/colab && pip install git+https://github.com/gbonnet1/notebooks.git


# %%
import agd.LinearParallel as lp
import matplotlib.pyplot as plt
import numpy as np
from agd import Domain, Selling
from agd.AutomaticDifferentiation import Dense2, Sparse
from agd.AutomaticDifferentiation.misc import tocsr
from agd.AutomaticDifferentiation.Optimization import newton_root
from scipy.sparse import diags
from scipy.sparse.linalg import eigs, spsolve


# %%
def domain(d):
    return Domain.Union(
        Domain.Ball(center=d * [0]),
        Domain.Box(sides=d * [(0, 1)]),
    )


# %%
def grid(d, h):
    return np.stack(
        np.meshgrid(
            *(d * [np.arange(-h * np.floor(1 / h), 1, h)]),
            indexing="ij",
        )
    )


# %%
def omega0(x):
    d = x.shape[0]

    if d == 2:
        return np.stack([np.cos(np.pi * x[1]), np.sin(np.pi * x[1])])
    elif d == 3:
        return np.stack(
            [
                np.cos(np.pi * x[1]),
                np.sin(np.pi * x[1]) * np.cos(np.pi * x[2]),
                np.sin(np.pi * x[1]) * np.sin(np.pi * x[2]),
            ]
        )
    else:
        raise ValueError(f"Invalid dimension: {d}")


def omega(x):
    return (2 - np.cos(np.pi * x[0])) / 3 * omega0(x)


def D(x):
    return (
        mu
        * (2 + np.cos(np.pi * x[0]))
        / 3
        * (
            nu * lp.identity(x.shape[1:])
            + (1 - nu) * lp.outer(omega0(x / 2), omega0(x / 2))
        )
    )


# %%
def u1(x):
    return 1 / 4 * lp.dot_VV(x, x) ** 2


def u2(x):
    return np.maximum(0, np.sqrt(lp.dot_VV(x, x)) - 0.4) ** 2.5


def u3(x):
    d = x.shape[0]
    return np.where(lp.dot_VV(x, x) < d, np.sqrt(d - lp.dot_VV(x, x)), 0)


# %%
def EqLinear(u_func, x):
    x_ad = Dense2.identity(constant=x, shape_free=x.shape[:1])
    u_ad = u_func(x_ad)
    du = np.moveaxis(u_ad.coef1, -1, 0)
    d2u = np.moveaxis(u_ad.coef2, [-2, -1], [0, 1])
    return -lp.dot_VV(omega(x), du) - lp.trace(lp.dot_AA(D(x), d2u))


def SchemeLinear(u, x, f, bc):
    coef, offsets = Selling.Decomposition(D(x))

    # coef_min = np.min(coef)
    # offsets_norm2 = lp.dot_VV(offsets, offsets)
    # offsets_max2 = np.max(np.where(coef < 1e-13, 0, offsets_norm2))
    # print(f"h: {bc.gridscale}, c: {coef_min}, e2: {offsets_max2}")

    du = bc.DiffCentered(u, offsets)
    d2u = bc.Diff2(u, offsets)
    return np.where(
        bc.interior,
        -lp.dot_VAV(omega(x), lp.inverse(D(x)), np.sum(coef * du * offsets, axis=1))
        - lp.dot_VV(coef, d2u)
        - f,
        u - bc.grid_values,
    )


def SolveLinear(x, f, bc):
    u = Sparse.identity(constant=np.zeros(x.shape[1:]))
    residue = SchemeLinear(u, x, f, bc)

    triplets, rhs = residue.solve(raw=True)
    mat = tocsr(triplets)

    if False:
        (val_max,), _ = eigs(mat, 1, which="LM")
        (val_min,), _ = eigs(mat, 1, which="SM")
        print(val_max / val_min)

    dde = (diags(mat.diagonal()) - mat).min() > -1e-8

    precond = diags(1 / mat.diagonal())
    matprecond = precond @ mat
    rhsprecond = precond @ rhs

    if False:
        (val_max,), _ = eigs(matprecond, 1, which="LM")
        (val_min,), _ = eigs(matprecond, 1, which="SM")
        print(val_max / val_min)

    return spsolve(matprecond, rhsprecond).reshape(x.shape[1:]), dde


# %%
def EqNonlinear(u_func, x):
    x_ad = Dense2.identity(constant=x, shape_free=x.shape[:1])
    u_ad = u_func(x_ad)
    du = np.moveaxis(u_ad.coef1, -1, 0)
    d2u = np.moveaxis(u_ad.coef2, [-2, -1], [0, 1])
    return -1 / 2 * lp.dot_VV(omega(x), du) ** 2 - lp.trace(lp.dot_AA(D(x), d2u))


def SchemeNonlinear(u, x, f, bc):
    coef, offsets = Selling.Decomposition(D(x))
    du = bc.DiffCentered(u, offsets)
    d2u = bc.Diff2(u, offsets)
    p = lp.dot_AV(lp.inverse(D(x)), np.sum(coef * du * offsets, axis=1))
    return np.where(
        bc.interior,
        -1 / 2 * lp.dot_VV(omega(x), p) ** 2 - lp.dot_VV(coef, d2u) - f,
        u - bc.grid_values,
    )


def SolveNonlinear(x, f, bc):
    dde = True

    def Solver(residue):
        nonlocal dde

        triplets, rhs = residue.solve(raw=True)
        mat = tocsr(triplets)

        # if (diags(mat.diagonal()) - mat).min() <= -1e-8:
        #     dde = False

        dde = (diags(mat.diagonal()) - mat).min() > -1e-8

        precond = diags(1 / mat.diagonal())
        matprecond = precond @ mat
        rhsprecond = precond @ rhs

        return spsolve(matprecond, rhsprecond).reshape(x.shape[1:])

    result = newton_root(
        SchemeNonlinear, 0.0001 * lp.dot_VV(x, x), params=(x, f, bc), solver=Solver
    )

    return result, dde


# %%
d = 2
mu = 1
nu = 1 / 10
h = 0.01

plt.figure(figsize=(9, 3))

for i, (u_func, title) in enumerate(
    [
        (u1, "Smooth function $u_1$"),
        (u2, "$C^{2, 0.5}$ function $u_2$"),
        (u3, "Singular function $u_3$"),
    ]
):
    x = grid(d, h)
    bc = Domain.Dirichlet(domain(d), u_func, x)

    u = u_func(x)
    f = EqLinear(u_func, x)

    u_approx, _ = SolveLinear(x, f, bc)

    plt.subplot(131 + i, aspect="equal")
    plt.title(title)
    im = plt.pcolormesh(*x, np.where(bc.interior, np.abs(u - u_approx), np.nan))
    plt.colorbar(im, orientation="horizontal", format="%.0e")

plt.savefig("linear-error-2d.png")
plt.show()


# %%
d = 2
mu = 1
nu = 1 / 10
h = 0.1 / 2 ** np.arange(0, 3.2, 0.2)

plt.figure(figsize=(9, 3))

for i, (u_func, title) in enumerate(
    [
        (u1, "Smooth function $u_1$"),
        (u2, "$C^{2, 0.5}$ function $u_2$"),
        (u3, "Singular function $u_3$"),
    ]
):
    err_l1 = np.zeros(h.shape)
    err_linf = np.zeros(h.shape)
    threshold = "any"

    plt.subplot(131 + i)
    plt.title(title)
    plt.xlabel("h")

    for j in range(len(h)):
        x = grid(d, h[j])
        bc = Domain.Dirichlet(domain(d), u_func, x)

        u = u_func(x)
        f = EqLinear(u_func, x)

        u_approx, dde = SolveLinear(x, f, bc)

        print(h[j], dde)

        if not dde:
            threshold = "none"
        elif threshold == "none":
            threshold = h[j]

        # if not dde:
        #     plt.axvspan(
        #         np.exp((np.log(h[min(len(h) - 1, j + 1)]) + np.log(h[j])) / 2),
        #         np.exp((np.log(h[j]) + np.log(h[max(0, j - 1)])) / 2),
        #         color="lightgray",
        #     )

        err_l1[j] = np.mean(np.abs(np.where(bc.interior, u - u_approx, 0)))
        err_linf[j] = np.max(np.abs(np.where(bc.interior, u - u_approx, 0)))

    print(f"*** threshold: {threshold} ***")

    plt.loglog(h, h / 16, "k:", label="order = 1")
    plt.loglog(h, h ** 2 / 16, "k--", label="order = 2")
    plt.loglog(h, err_linf, ".-", label="$l^\infty$ error")
    plt.loglog(h, err_l1, ".-", label="$l^1$ error")

    if i == 0:
        plt.legend()
    plt.xticks(rotation=30)
    for text in plt.gca().get_xminorticklabels():
        text.set_rotation(30)

plt.tight_layout()
plt.savefig("linear-convergence-2d.png")
plt.show()


# %%
d = 2
mu = 2
nu = 1 / 10
h = 0.1 / 2 ** np.arange(0, 3.2, 0.2)

plt.figure(figsize=(9, 3))

for i, (u_func, title) in enumerate(
    [
        (u1, "Smooth function $u_1$"),
        (u2, "$C^{2, 0.5}$ function $u_2$"),
        (u3, "Singular function $u_3$"),
    ]
):
    err_l1 = np.zeros(h.shape)
    err_linf = np.zeros(h.shape)
    threshold = "any"

    plt.subplot(131 + i)
    plt.title(title)
    plt.xlabel("h")

    for j in range(len(h)):
        x = grid(d, h[j])
        bc = Domain.Dirichlet(domain(d), u_func, x)

        u = u_func(x)
        f = EqNonlinear(u_func, x)

        u_approx, dde = SolveNonlinear(x, f, bc)

        print(h[j], dde)

        if not dde:
            threshold = "none"
        elif threshold == "none":
            threshold = h[j]

        # if not dde:
        #     plt.axvspan(
        #         np.exp((np.log(h[min(len(h) - 1, j + 1)]) + np.log(h[j])) / 2),
        #         np.exp((np.log(h[j]) + np.log(h[max(0, j - 1)])) / 2),
        #         color="lightgray",
        #     )

        err_l1[j] = np.mean(np.abs(np.where(bc.interior, u - u_approx, 0)))
        err_linf[j] = np.max(np.abs(np.where(bc.interior, u - u_approx, 0)))

    print(f"*** threshold: {threshold} ***")

    plt.loglog(h, h / 16, "k:", label="order = 1")
    plt.loglog(h, h ** 2 / 16, "k--", label="order = 2")
    plt.loglog(h, err_linf, ".-", label="$l^\infty$ error")
    plt.loglog(h, err_l1, ".-", label="$l^1$ error")

    # if i == 0:
    #     plt.legend()
    plt.xticks(rotation=30)
    for text in plt.gca().get_xminorticklabels():
        text.set_rotation(30)

plt.tight_layout()
plt.savefig("nonlinear-convergence-2d.png")
plt.show()


# %%
d = 3
mu = 4
nu = 1 / 10
h = 0.3 / 2 ** np.arange(0, 2.2, 0.2)

plt.figure(figsize=(9, 3))

for i, (u_func, title) in enumerate(
    [
        (u1, "Smooth function $u_1$"),
        (u2, "$C^{2, 0.5}$ function $u_2$"),
        (u3, "Singular function $u_3$"),
    ]
):
    err_l1 = np.zeros(h.shape)
    err_linf = np.zeros(h.shape)
    threshold = "any"

    plt.subplot(131 + i)
    plt.title(title)
    plt.xlabel("h")

    for j in range(len(h)):
        x = grid(d, h[j])
        bc = Domain.Dirichlet(domain(d), u_func, x)

        u = u_func(x)
        f = EqLinear(u_func, x)

        u_approx, dde = SolveLinear(x, f, bc)

        print(h[j], dde)

        if not dde:
            threshold = "none"
        elif threshold == "none":
            threshold = h[j]

        # if not dde:
        #     plt.axvspan(
        #         np.exp((np.log(h[min(len(h) - 1, j + 1)]) + np.log(h[j])) / 2),
        #         np.exp((np.log(h[j]) + np.log(h[max(0, j - 1)])) / 2),
        #         color="lightgray",
        #     )

        err_l1[j] = np.mean(np.abs(np.where(bc.interior, u - u_approx, 0)))
        err_linf[j] = np.max(np.abs(np.where(bc.interior, u - u_approx, 0)))

    print(f"*** threshold: {threshold} ***")

    plt.loglog(h, h / 16, "k:", label="order = 1")
    plt.loglog(h, h ** 2 / 16, "k--", label="order = 2")
    plt.loglog(h, err_linf, ".-", label="$l^\infty$ error")
    plt.loglog(h, err_l1, ".-", label="$l^1$ error")

    # if i == 0:
    #     plt.legend()
    plt.xticks(rotation=30)
    for text in plt.gca().get_xminorticklabels():
        text.set_rotation(30)

plt.tight_layout()
plt.savefig("linear-convergence-3d.png")
plt.show()


# %%
d = 3
mu = 8
nu = 1 / 10
h = 0.3 / 2 ** np.arange(0, 2.2, 0.2)

plt.figure(figsize=(9, 3))

for i, (u_func, title) in enumerate(
    [
        (u1, "Smooth function $u_1$"),
        (u2, "$C^{2, 0.5}$ function $u_2$"),
        (u3, "Singular function $u_3$"),
    ]
):
    err_l1 = np.zeros(h.shape)
    err_linf = np.zeros(h.shape)
    threshold = "any"

    plt.subplot(131 + i)
    plt.title(title)
    plt.xlabel("h")

    for j in range(len(h)):
        x = grid(d, h[j])
        bc = Domain.Dirichlet(domain(d), u_func, x)

        u = u_func(x)
        f = EqNonlinear(u_func, x)

        u_approx, dde = SolveNonlinear(x, f, bc)

        print(h[j], dde)

        if not dde:
            threshold = "none"
        elif threshold == "none":
            threshold = h[j]

        # if not dde:
        #     plt.axvspan(
        #         np.exp((np.log(h[min(len(h) - 1, j + 1)]) + np.log(h[j])) / 2),
        #         np.exp((np.log(h[j]) + np.log(h[max(0, j - 1)])) / 2),
        #         color="lightgray",
        #     )

        err_l1[j] = np.mean(np.abs(np.where(bc.interior, u - u_approx, 0)))
        err_linf[j] = np.max(np.abs(np.where(bc.interior, u - u_approx, 0)))

    print(f"*** threshold: {threshold} ***")

    plt.loglog(h, h / 16, "k:", label="order = 1")
    plt.loglog(h, h ** 2 / 16, "k--", label="order = 2")
    plt.loglog(h, err_linf, ".-", label="$l^\infty$ error")
    plt.loglog(h, err_l1, ".-", label="$l^1$ error")

    # if i == 0:
    #     plt.legend()
    plt.xticks(rotation=30)
    for text in plt.gca().get_xminorticklabels():
        text.set_rotation(30)

plt.tight_layout()
plt.savefig("nonlinear-convergence-3d.png")
plt.show()


# %%
