<!--hopefully whole SLAM in rust, but for now frontend and visual odometry.-->
# SLAM-RS

Trying to implement a SLAM system in Rust.

## Dependencies

- [OpenCV](https://crates.io/crates/opencv)
- [nalgebra](https://crates.io/crates/nalgebra)
- [bevy](https://crates.io/crates/bevy)
- [serde](https://crates.io/crates/serde)

## Usage

Build with Cargo:

```
cargo build --release
```

Run visual odometry example:

```
cargo run --example visual_odometry -- test.mp4
```

## Modules

- `feature`: ORB feature detection and matching
- `odometry`: Camera intrinsics, pose estimation, trajectory tracking

See [TODO](TODO.md) for development status.
