<!--hopefully whole SLAM in rust, but for now frontend and visual odometry.-->
# SLAM-RS

Trying to implement a SLAM system in Rust.

## Dependencies

- [OpenCV](https://crates.io/crates/opencv)
- [nalgebra](https://crates.io/crates/nalgebra)
- [serde](https://crates.io/crates/serde)
- [rerun](https://crates.io/crates/rerun) (optional, for 3D visualization)
- [Clap](https://crates.io/crates/clap) (for command-line argument parsing)
- [tch-rs](https://crates.io/crates/tch) (optional, for deep learning depth estimation)

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

Run monocular depth estimation:

```bash
# Single image
cargo run --example depth_estimation --features depth -- test.jpg

# Video with Rerun visualization (requires model files in weights/)
cargo run --example depth_estimation --features depth,rerun -- test.mp4 --cuda --rerun

# See docs/Deep-Learning.md for model setup
```

## Modules

- `feature`: ORB feature detection and matching
- `odometry`: Camera intrinsics, pose estimation, trajectory tracking
- `mapping`: Keyframe selection, 3D point triangulation, map points, bundle adjustment
- `depth`: Monocular depth estimation with MonoDepth2 (optional, requires `depth` feature)

## Examples

- `visualize_features`: Real-time feature detection and matching visualization
- `visual_odometry`: Full VO pipeline with trajectory tracking and visualization
- `point_cloud`: 3D point cloud reconstruction with triangulation and bundle adjustment
- `bundle_adjustment`: Demonstration of pose and point optimization
- `depth_estimation`: Monocular depth estimation with MonoDepth2 (requires `--features depth`)

See [TODO](TODO.md) for development status and [docs/Deep-Learning.md](docs/Deep-Learning.md) for depth estimation setup.

## FAQ

- **Why is the map sparse?**
  - This is a feature-based system using ORB features (~1000-2000 per frame). We only triangulate at corner-like features, not every pixel. This is similar to ORB-SLAM.
- **How can I make the map denser?**
  - Use the MonoDepth2 depth estimation module (see `docs/Deep-Learning.md`)
  - For dense reconstruction, you need:
    - Dense/semi-dense tracking (all high-gradient pixels)
    - Depth estimation/fusion
    - More computational resources

## Features

| Feature | Status | Cargo Flag |
|---------|--------|------------|
| Feature-based VO | ✅ | (default) |
| Bundle Adjustment | ✅ | (default) |
| Rerun Visualization | ✅ | `--features rerun` |
| Depth Estimation | ✅ | `--features depth` |
| Loop Closure | 🚧 | - |
