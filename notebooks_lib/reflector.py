import subprocess
import tempfile

import imageio
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

header = """#version 3.7;

#include "colors.inc"
"""

footer = """
global_settings {
    assumed_gamma 1.0

    ambient_light 0

    photons {
        count 1000000
    }
}

camera {
    orthographic
    location <0, 1, 0>
    sky <0, 0, 1>
    right <0, 0, 2>
    up <2, 0, 0>
    look_at  <0, 0,  0>
}

light_source {
    <0, 1, 0>,
    color srgb <0.5, 0.5, 0.5>
    cylinder
    radius 1
    point_at <0, 2, 0>
}

mesh2 {
    vertex_vectors {
        4,
        <-1, 0, -1>,
        <1, 0, -1>,
        <1, 0, 1>,
        <-1, 0, 1>
    }
    face_indices {
        2,
        <0, 1, 2>,
        <0, 2, 3>
    }
    texture {
        pigment { color White }
    }
}

object {
    Reflector
    texture {
        pigment { color White }
        finish {
            reflection 1
        }
    }
    photons {
        target
        reflection on
    }
}"""


def simulate_reflector(y, z):
    tri = Triangulation(*y)

    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        filename = f.name

        print(header, file=f)
        print("#declare Reflector = mesh2 {", file=f)
        print("    vertex_vectors {", file=f)
        print(f"        {y.shape[1]}", end="", file=f)
        for i in range(y.shape[1]):
            print(",", file=f)
            print(f"        <{y[0, i]}, {z[i]}, {y[1, i]}>", end="", file=f)
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
        print(footer, file=f)

    result = subprocess.run(
        ["povray", f"+I{filename}", "+O-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    im = imageio.imread(result.stdout)
    plt.imshow(im)
    plt.show()
