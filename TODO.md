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
- [x] Implement keyframe selection (translation, rotation, match quality criteria)
- [x] Add 3D point triangulation/mapping
- [x] Create MapPoint struct for 3D points
- [x] Implement Triangulator for computing 3D points from 2D correspondences
- [x] Add point cloud export (PLY and JSON formats)
- [x] Create point cloud example with triangulation
- [x] Add real-time 3D visualization with Rerun

## Current Issues
- [ ] Point cloud is sparse - only ~1000-2000 ORB features per frame
- [ ] No map management - points are not reobserved, no global map consistency
- [ ] This is VO, not SLAM - missing loop closure and global optimization
- [ ] Scale drift - no absolute scale, accumulates error over time
- [ ] No visualization - hard to see what's being mapped

## SLAM Roadmap (Priority Order)

### Basic SLAM Infrastructure (Current Focus)
- [x] Visual odometry (camera tracking)
- [x] Sparse 3D reconstruction (triangulation)
- [x] Real-time visualization with Rerun - see what's being mapped!
- [ ] Map management - track which points exist, deduplicate, prune outliers
- [ ] Point reobservation - match against existing map points, not just previous frame
- [ ] Local mapping - maintain sliding window of recent keyframes and points

### Dense/Semi-Dense Reconstruction
- [ ] Increase point density
  - [ ] Use SIFT/SURF (more features than ORB)
  - [ ] Semi-dense tracking (high gradient pixels, not just corners)
  - [ ] Depth map estimation between keyframes
- [ ] Depth filtering - probabilistic depth estimation for each pixel
- [ ] Depth fusion - merge depth estimates from multiple views

### Loop Closure & Global Optimization
- [ ] Place recognition - DBoW2/DBoW3 for detecting revisited locations
- [ ] Loop closure detection - geometric verification of loop candidates
- [ ] Pose graph optimization - correct drift when loop is detected (g2o/Ceres)
- [ ] Global bundle adjustment - optimize all poses and points together

### Robustness & Production
- [ ] Relocalization - recover from tracking loss
- [ ] Map saving/loading - persist maps between runs
- [ ] Multi-threading** - separate tracking, local mapping, loop closing threads
- [ ] IMU integration - use IMU for better tracking (VI-SLAM)
- [ ] Camera calibration module - estimate intrinsics from video

### Phase Future
- [ ] Support stereo cameras (true scale from stereo baseline)
- [ ] Support RGB-D cameras (direct depth from sensor)
- [ ] Object detection integration
- [ ] Semantic SLAM (label 3D points with objects)
- [ ] Neural depth estimation (monocular depth networks)

## Technical Debt
- [ ] Handle degenerate cases (insufficient matches, pure rotation, etc.)
- [ ] Better error handling throughout
- [ ] More comprehensive tests
- [ ] Benchmark on KITTI dataset with ground truth comparison
- [ ] GPU acceleration (feature detection, matching)
- [ ] Add more camera presets

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

**Run point cloud generation with triangulation:**
```bash
# With Rerun 3D viewer (shows map, trajectory, matches, video in real-time!)
cargo run --example point_cloud --features rerun /path/to/video.mp4 -- --rerun

# Or save to PLY file (default, no Rerun)
cargo run --example point_cloud /path/to/video.mp4 -- --save-ply

# With custom camera intrinsics
cargo run --example point_cloud --features rerun /path/to/video.mp4 -- --rerun --fx 718.856 --fy 718.856 --cx 607.1928 --cy 185.2157
```


**Run tests:**
```bash
cargo test
```

**Run main:**
```bash
cargo run -- /path/to/video.mp4
```
