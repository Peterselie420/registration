import os
import time
import reglib


def main():
    # Only needed if you want to use manually compiled library code
    # reglib.load_library(os.path.join(os.curdir, "cmake-build-debug"))

    # Load you data
    source_points = reglib.load_data(os.path.join(os.curdir, "files", "single.ply"))
    target_points = reglib.load_data(os.path.join(os.curdir, "files", "full.ply"))

    # Run the registration algorithm
    start = time.time()
    trans = reglib.icp(source=source_points, target=target_points, nr_iterations=10, epsilon=0.01,
                       inlier_threshold=0.05, distance_threshold=5.0, downsample=0, visualize=True)
                       #resolution=12.0, step_size=0.5, voxelize=0)
    print("Runtime:", time.time() - start)
    print(trans)


if __name__ == "__main__":
    main()
