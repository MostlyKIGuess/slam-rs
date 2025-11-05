# TODO

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
- [ ] Compute essential matrix
- [ ] Recover pose (R, t)
- [ ] Build transformation matrix
- [ ] Update global pose
- [ ] Track trajectory
- [ ] Save trajectory to JSON
- [ ] Add 3D visualization
- [ ] Add CLI arguments
- [ ] Handle errors gracefully
- [ ] Write tests
- [ ] Document code
- [ ] Test on KITTI dataset

## How to Run

**Run the visualizer:**
```bash
cargo run --example visualize_features /path/to/video.mp4
```

**Run tests:**
```bash
cargo test
```

**Run main:**
```bash
cargo run -- /path/to/video.mp4
```