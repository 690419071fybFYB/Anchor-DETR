import numpy as np
import matplotlib.pyplot as plt


def main():
    x = np.linspace(-10, 10, 200)
    y = np.linspace(-10, 10, 200)
    xx, yy = np.meshgrid(x, y)
    zz = xx * (2 * yy - 1) - 16

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, zz, cmap='viridis', edgecolor='none', alpha=0.8)
    ax.set_title(r"$z = x(2y - 1) - 16$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.tight_layout()
    plt.savefig("z_surface.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
