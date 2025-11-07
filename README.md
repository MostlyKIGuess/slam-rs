<!--hopefully whole SLAM in rust, but for now frontend and visual odometry.-->
# SLAM-RS

Trying to implement a SLAM system in Rust.

## Dependencies

- [OpenCV](https://crates.io/crates/opencv)
- [nalgebra](https://crates.io/crates/nalgebra)
- [serde](https://crates.io/crates/serde)
- [rerun](https://crates.io/crates/rerun) (optional, for 3D visualization)
- [Clap](https://crates.io/crates/clap) (for command-line argument parsing)

## Usage

Build with Cargo:

```
cargo build --release
```

Run visual odometry example:

```
# Use default KITTI intrinsics
cargo run --example visual_odometry /path/to/video.mp4

# Specify custom camera intrinsics
cargo run --example visual_odometry /path/to/video.mp4 -- --fx 500 --fy 500 --cx 320 --cy 240
```

Run point cloud generation with real-time 3D visualization (Rerun):

```bash
# With Rerun 3D viewer (shows map, trajectory, matches, video in real-time!)
cargo run --example point_cloud --features rerun /path/to/video.mp4 -- --rerun

# Or save to PLY file (default, no Rerun)
cargo run --example point_cloud /path/to/video.mp4 -- --save-ply

# With custom camera intrinsics
cargo run --example point_cloud --features rerun /path/to/video.mp4 -- --rerun --fx 718.856 --fy 718.856 --cx 607.1928 --cy 185.2157
```

Run feature detection visualization:

```bash
cargo run --example visualize_features /path/to/video.mp4
```

## Modules

- `feature`: ORB feature detection and matching
- `odometry`: Camera intrinsics, pose estimation, trajectory tracking
- `mapping`: Keyframe selection, 3D point triangulation, map points

## Examples

- `visualize_features`: Real-time feature detection and matching visualization
- `visual_odometry`: Full VO pipeline with trajectory tracking and visualization
- `point_cloud`: 3D point cloud reconstruction with triangulation (use `--features rerun --rerun` for rerun viz!)

See [TODO](TODO.md) for development status.

## FAQ

- **Why is the map sparse?**
  - This is a feature-based system using ORB features (~1000-2000 per frame). We only triangulate at corner-like features, not every pixel. This is similar to ORB-SLAM.
- **How can I make the map denser?**
  - For dense reconstruction, you need:
    - Dense/semi-dense tracking (all high-gradient pixels)
    - Depth estimation/fusion
    - More computational resources
