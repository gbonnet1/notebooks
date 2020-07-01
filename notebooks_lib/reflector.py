import subprocess
import tempfile

import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation
from scipy.interpolate import griddata

header = """#version 3.7;

#include "colors.inc"
"""


def footer(intensity, target_radius):
    return f"""
global_settings {{
    assumed_gamma 1.0

    ambient_light 0

    photons {{
        count 1000000
    }}
}}

camera {{
    orthographic
    location <0, 0, -1>
    right <{2 * target_radius}, 0, 0>
    up <0, {2 * target_radius}, 0>
    look_at  <0, 0, 0>
}}

light_source {{
    <0, 0, 0>,
    color srgb <{intensity}, {intensity}, {intensity}>
    cylinder
    radius 1
    point_at <0, 0, -1>
}}

mesh2 {{
    vertex_vectors {{
        4,
        <{-target_radius}, {target_radius}, 0>,
        <{-target_radius}, {-target_radius}, 0>,
        <{target_radius}, {-target_radius}, 0>,
        <{target_radius}, {target_radius}, 0>
    }}
    face_indices {{
        2,
        <0, 1, 2>,
        <0, 2, 3>
    }}
    texture {{
        pigment {{ color White }}
    }}
}}

object {{
    Reflector
    texture {{
        pigment {{ color White }}
        finish {{
            reflection 1
        }}
    }}
    photons {{
        target
        reflection on
    }}
}}"""


def simulate_reflector(y, z, intensity=0.5, target_radius=1):
    x = np.stack(np.meshgrid(*(2 * [np.linspace(-1, 1, 500)]), indexing="ij"))
    v = griddata((y[0], y[1]), z, (x[0], x[1]), method="cubic")

    x = x[:, np.logical_not(np.isnan(v))]
    v = v[np.logical_not(np.isnan(v))]

    tri = Triangulation(*x)

    with tempfile.NamedTemporaryFile("w", suffix=".pov", delete=False) as f:
        infile = f.name

        print(header, file=f)
        print("#declare Reflector = mesh2 {", file=f)
        print("    vertex_vectors {", file=f)
        print(f"        {x.shape[1]}", end="", file=f)
        for i in range(x.shape[1]):
            print(",", file=f)
            print(f"        <{x[0, i]}, {x[1, i]}, {-v[i]}>", end="", file=f)
        print("", file=f)
        print("    }", file=f)
        print("    face_indices {", file=f)
        print(f"        {len(tri.triangles)}", end="", file=f)
        for n in tri.triangles:
            print(",", file=f)
            print(f"        <{n[0]}, {n[1]}, {n[2]}>", end="", file=f)
        print("", file=f)
        print("    }", file=f)
        print("};", file=f)
        print(footer(intensity, target_radius), file=f)

    with tempfile.NamedTemporaryFile("w", suffix=".png", delete=False) as f:
        outfile = f.name

    subprocess.run(
        ["povray", f"+I{infile}", f"+O{outfile}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )

    im = imageio.imread(outfile)

    plt.imshow(im)
    plt.show()
