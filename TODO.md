# TODO

## Completed
- [x] Setup project structure
- [x] Add dependencies (OpenCV, nalgebra, bevy, serde)
- [x] Read video frames from file
- [x] Convert frames to grayscale
- [x] Detect ORB features
- [x] Create OrbDetector wrapper (modularized in src/feature/)
- [x] Write basic tests for feature detection
- [x] Match features between frames
- [x] Create FeatureMatcher with BFMatcher
- [x] Filter matches by distance
- [x] Create visualizer example
- [x] Extract matched point coordinates (Point2f)
- [x] Create CameraIntrinsics struct
- [x] Compute essential matrix from point pairs
- [x] Recover pose (R, t) from essential matrix
- [x] Convert OpenCV Mat to nalgebra types
- [x] Build 4x4 transformation matrix
- [x] Initialize global pose tracking
- [x] Update global pose (compose transformations)
- [x] Store trajectory points
- [x] Calculate total distance traveled
- [x] Save trajectory to JSON file
- [x] Add unit tests for pose recovery
- [x] Add example for full visual odometry pipeline with trajectory visualization

## In Progress / TODO
- [ ] Add camera calibration module
- [ ] Handle degenerate cases (insufficient matches)
- [ ] Document code ( I started doing this, will keep consistent /// docstrings)
- [x] Test on sample video
- [ ] Test on KITTI dataset
- [ ] Add 3D visualization with Bevy
- [ ] Optimize performance (threading, GPU)
- [ ] Add loop closure detection
- [ ] Implement bundle adjustment
- [ ] Add more camera presets
- [ ] Support stereo cameras

## Suggest TODO?

- Contributor? Please submit a pull request or add a TODO here with a ticket.


## How to Run

**Run feature visualizer:**
```bash
cargo run --example visualize_features /path/to/video.mp4
```

**Run full visual odometry with trajectory:**
```bash
# Use default KITTI intrinsics
cargo run --example visual_odometry /path/to/video.mp4

# Specify custom camera intrinsics
cargo run --example visual_odometry /path/to/video.mp4 -- --fx 500 --fy 500 --cx 320 --cy 240
```

**Run tests:**
```bash
cargo test
```

**Run main:**
```bash
cargo run -- /path/to/video.mp4
```
