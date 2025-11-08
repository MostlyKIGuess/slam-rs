use opencv::{
    core::{KeyPoint, Mat, Point2f, Scalar, Vector},
    features2d, highgui, imgproc,
    prelude::*,
    videoio::{self, VideoCapture, VideoCaptureTrait},
};
use slamkit_rs::{
    BundleAdjuster, CameraIntrinsics, FeatureMatcher, KeyframeConfig, KeyframeSelector, Map,
    MapPoint, Observation, OrbDetector, PoseEstimator, Trajectory, Triangulator,
};
use std::fs::File;
use std::io::Write;
use std::time::Instant;

use clap::Parser;

#[cfg(feature = "rerun")]
use rerun::{RecordingStreamBuilder, external::glam};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    #[arg(index = 1)]
    video_path: String,
    #[arg(long)]
    fx: Option<f64>,
    #[arg(long)]
    fy: Option<f64>,
    #[arg(long)]
    cx: Option<f64>,
    #[arg(long)]
    cy: Option<f64>,
    #[arg(long)]
    rerun: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Point Cloud Generation with Triangulation");

    let cli = Cli::parse();

    let intrinsics =
        if let (Some(fx), Some(fy), Some(cx), Some(cy)) = (cli.fx, cli.fy, cli.cx, cli.cy) {
            println!("Using provided camera intrinsics:");
            CameraIntrinsics::new(fx, fy, cx, cy)
        } else {
            println!("No intrinsics provided. Using KITTI defaults.");
            CameraIntrinsics::kitti()
        };

    let video_path = &cli.video_path;
    let use_rerun = cli.rerun;

    println!("  fx: {:.2}, fy: {:.2}", intrinsics.fx, intrinsics.fy);
    println!("  cx: {:.2}, cy: {:.2}", intrinsics.cx, intrinsics.cy);
    println!(
        "Visualization: {}\n",
        if use_rerun { "Rerun" } else { "OpenCV" }
    );

    #[cfg(feature = "rerun")]
    let rec = if use_rerun {
        // Use spawn() for a live view
        Some(RecordingStreamBuilder::new("slam-rs").spawn()?)

        // Use save() to write to a file first
        // println!("Saving to slam_output.rrd... Run 'rerun slam_output.rrd' to view.");
        // Some(RecordingStreamBuilder::new("slam-rs").save("slam_output.rrd")?)
    } else {
        None
    };

    #[cfg(not(feature = "rerun"))]
    if use_rerun {
        eprintln!("Warning: Rerun not available. Build with --features rerun");
        eprintln!("Falling back to OpenCV visualization.\n");
    }

    println!("Opening video: {}", video_path);
    let mut cap = VideoCapture::from_file(&video_path, videoio::CAP_ANY)?;

    if !cap.is_opened()? {
        return Err("Cannot open video".into());
    }

    let width = cap.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32;
    let height = cap.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32;
    let fps = cap.get(videoio::CAP_PROP_FPS)?;

    println!("Resolution: {}x{}", width, height);
    println!("FPS: {:.2}\n", fps);

    // Increase features for denser map
    // This isn't working
    // Need to improve the whole pipeline and have to come back later here
    // TODO^
    let mut detector = OrbDetector::new(3000)?;
    let mut matcher = FeatureMatcher::new()?;
    let pose_estimator = PoseEstimator::new(intrinsics);
    let triangulator = Triangulator::new(intrinsics)
        .with_min_parallax(0.5)
        .with_max_reproj_error(8.0);
    let mut trajectory = Trajectory::new();

    // Bundle adjustment for optimization
    let bundle_adjuster = BundleAdjuster::new(intrinsics)
        .with_max_iterations(10)
        .with_lambda(1e-3);
    let mut all_observations: Vec<Observation> = Vec::new();
    let mut ba_runs = 0;

    // More aggressive keyframe selection for more points
    let kf_config = KeyframeConfig {
        min_translation: 0.03,
        min_rotation: 0.03,
        min_match_ratio: 0.7,
        max_frames: 3,
    };
    let mut keyframe_selector = KeyframeSelector::with_config(kf_config);

    // Global map for point management and reobservation
    let mut global_map = Map::new(intrinsics);

    if !use_rerun {
        highgui::named_window("Video", highgui::WINDOW_AUTOSIZE)?;
        highgui::named_window("Matches", highgui::WINDOW_AUTOSIZE)?;
        highgui::named_window("Trajectory", highgui::WINDOW_AUTOSIZE)?;
        highgui::named_window("3D Map", highgui::WINDOW_AUTOSIZE)?;
    }

    let mut frame = Mat::default();
    let mut gray = Mat::default();
    let mut prev_gray = Mat::default();
    let mut prev_kp = Vector::new();
    let mut prev_desc = Mat::default();
    let mut frame_count = 0;
    let mut keyframe_count = 0;

    let mut prev_keyframe_pose: Option<(nalgebra::Matrix3<f64>, nalgebra::Vector3<f64>)> = None;
    let mut prev_keyframe_kp = Vector::new();
    let mut prev_keyframe_desc = Mat::default();
    let mut prev_keyframe_gray = Mat::default(); // For Rerun match image

    let start_time = Instant::now();

    #[cfg(feature = "rerun")]
    if let Some(ref rec) = rec {
        // Log static coordinate system for the world
        rec.log_static("world", &rerun::ViewCoordinates::RUB())?;
    }

    println!("Processing... Press 'q' to quit, 's' to save\n");

    loop {
        if !cap.read(&mut frame)? || frame.empty() {
            break;
        }

        frame_count += 1;

        imgproc::cvt_color(
            &frame,
            &mut gray,
            imgproc::COLOR_BGR2GRAY,
            0,
            opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        #[cfg(feature = "rerun")]
        if let Some(ref rec) = rec {
            if frame_count % 3 == 0 {
                // Set the time for this frame
                rec.set_time_sequence("frame", frame_count as i64);
                log_frame_to_rerun(rec, &frame)?;
            }
        }

        let (kp, desc) = detector.detect_and_compute(&gray)?;

        if frame_count > 1 && !prev_desc.empty() {
            let matches = matcher.match_descriptors(&prev_desc, &desc)?;
            let good_matches = matcher.filter_good_matches(&matches, 2.0);

            if good_matches.len() >= 8 {
                let (points1, points2) =
                    pose_estimator.extract_matched_points(&prev_kp, &kp, &good_matches)?;

                match pose_estimator.compute_essential_matrix(&points1, &points2) {
                    Ok(essential) => {
                        match pose_estimator.recover_pose(&essential, &points1, &points2) {
                            Ok((rotation, translation)) => {
                                let is_keyframe = keyframe_selector.should_be_keyframe(
                                    &rotation,
                                    &translation,
                                    good_matches.len(),
                                );

                                if is_keyframe {
                                    keyframe_count += 1;
                                    let timestamp = (frame_count - 1) as f64 / fps;
                                    trajectory.update(
                                        &rotation,
                                        &translation,
                                        frame_count as usize,
                                        timestamp,
                                    );

                                    let current_pose_rt = trajectory.current_pose_rt();

                                    #[cfg(feature = "rerun")]
                                    if let Some(ref rec) = rec {
                                        // Set time for the camera pose
                                        rec.set_time_sequence("frame", frame_count as i64);
                                        log_camera_pose(rec, &current_pose_rt)?;
                                    }

                                    // Triangulate between previous keyframe and current
                                    if let Some(prev_pose) = &prev_keyframe_pose {
                                        if !prev_keyframe_desc.empty() {
                                            let kf_matches = matcher
                                                .match_descriptors(&prev_keyframe_desc, &desc)?;
                                            let good_kf_matches =
                                                matcher.filter_good_matches(&kf_matches, 2.0);

                                            if good_kf_matches.len() >= 8 {
                                                let (kf_points1, kf_points2) = pose_estimator
                                                    .extract_matched_points(
                                                        &prev_keyframe_kp,
                                                        &kp,
                                                        &good_kf_matches,
                                                    )?;

                                                #[cfg(feature = "rerun")]
                                                if let Some(ref rec) = rec {
                                                    // Set time for the 2D matches
                                                    rec.set_time_sequence(
                                                        "frame",
                                                        frame_count as i64,
                                                    );

                                                    // Log points overlay
                                                    log_matches_to_rerun(
                                                        rec,
                                                        &kf_points1,
                                                        &kf_points2,
                                                    )?;

                                                    // Log separate side-by-side match image
                                                    if !prev_keyframe_gray.empty() {
                                                        log_matches_image_to_rerun(
                                                            rec,
                                                            &prev_keyframe_gray,
                                                            &gray,
                                                            &prev_keyframe_kp,
                                                            &kp,
                                                            &good_kf_matches,
                                                        )?;
                                                    }
                                                }

                                                match triangulator.triangulate(
                                                    prev_pose,
                                                    &current_pose_rt,
                                                    &kf_points1,
                                                    &kf_points2,
                                                    Some(&desc),
                                                ) {
                                                    Ok(points) => {
                                                        let num_triangulated = points.len();

                                                        // Transform points to world frame
                                                        let world_points =
                                                            transform_points_to_world(
                                                                &points,
                                                                &current_pose_rt,
                                                            );

                                                        // Add new points to global map
                                                        global_map.add_points(world_points);

                                                        // Try to match current frame against existing map
                                                        if let Ok(matches) = global_map
                                                            .find_matches(
                                                                &kp,
                                                                &desc,
                                                                &current_pose_rt,
                                                                &mut matcher,
                                                            )
                                                        {
                                                            global_map
                                                                .update_observations(&matches);

                                                            // Collect observations for BA
                                                            for (map_id, kp_idx) in matches.iter() {
                                                                let pixel = kp.get(*kp_idx)?.pt();
                                                                all_observations.push(
                                                                    Observation::new(
                                                                        keyframe_count - 1,
                                                                        *map_id,
                                                                        nalgebra::Point2::new(
                                                                            pixel.x as f64,
                                                                            pixel.y as f64,
                                                                        ),
                                                                    ),
                                                                );
                                                            }
                                                        }

                                                        // Run bundle adjustment every 5 keyframes
                                                        if keyframe_count % 5 == 0
                                                            && keyframe_count > 1
                                                        {
                                                            println!(
                                                                "  Running Bundle Adjustment..."
                                                            );
                                                            let mut poses_vec: Vec<(
                                                                nalgebra::Matrix3<f64>,
                                                                nalgebra::Vector3<f64>,
                                                            )> = trajectory
                                                                .points()
                                                                .iter()
                                                                .map(|tp| {
                                                                    (
                                                                        nalgebra::Matrix3::identity(
                                                                        ),
                                                                        nalgebra::Vector3::new(
                                                                            tp.position[0],
                                                                            tp.position[1],
                                                                            tp.position[2],
                                                                        ),
                                                                    )
                                                                })
                                                                .collect();

                                                            let mut points_vec: Vec<
                                                                nalgebra::Point3<f64>,
                                                            > = global_map
                                                                .points()
                                                                .iter()
                                                                .map(|p| p.position)
                                                                .collect();

                                                            if poses_vec.len() > 1
                                                                && points_vec.len() > 3
                                                                && all_observations.len() > 10
                                                            {
                                                                match bundle_adjuster
                                                                    .local_bundle_adjustment(
                                                                        &mut poses_vec,
                                                                        &mut points_vec,
                                                                        &all_observations,
                                                                        5,
                                                                    ) {
                                                                    Ok(error) => {
                                                                        ba_runs += 1;
                                                                        println!(
                                                                            "  BA: Optimized {} poses, {} points. Error: {:.4}",
                                                                            poses_vec.len(),
                                                                            points_vec.len(),
                                                                            error
                                                                        );
                                                                    }
                                                                    Err(e) => {
                                                                        eprintln!(
                                                                            "  BA failed: {}",
                                                                            e
                                                                        );
                                                                    }
                                                                }
                                                            }
                                                        }

                                                        // Prune outliers periodically
                                                        if keyframe_count % 10 == 0 {
                                                            let removed =
                                                                global_map.prune_outliers();
                                                            if removed > 0 {
                                                                println!(
                                                                    "  Pruned {} outlier points",
                                                                    removed
                                                                );
                                                            }
                                                        }

                                                        #[cfg(feature = "rerun")]
                                                        if let Some(ref rec) = rec {
                                                            let all_points: Vec<MapPoint> =
                                                                global_map
                                                                    .points()
                                                                    .iter()
                                                                    .map(|p| (*p).clone())
                                                                    .collect();
                                                            log_map_points(rec, &all_points)?;
                                                            log_trajectory(rec, &trajectory)?;
                                                        }

                                                        println!(
                                                            "Frame {:4} | KF {:3} | Matches: {:3} | Tri: {:4} | Map: {:6} ({:4} stable)",
                                                            frame_count,
                                                            keyframe_count,
                                                            good_kf_matches.len(),
                                                            num_triangulated,
                                                            global_map.size(),
                                                            global_map.stable_points().len()
                                                        );
                                                    }
                                                    Err(e) => {
                                                        eprintln!("Triangulation error: {}", e);
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    prev_keyframe_pose = Some(current_pose_rt);
                                    prev_keyframe_kp = kp.clone();
                                    desc.copy_to(&mut prev_keyframe_desc)?;
                                    gray.copy_to(&mut prev_keyframe_gray)?;
                                }
                            }
                            Err(_) => {}
                        }
                    }
                    Err(_) => {}
                }

                if !use_rerun {
                    let mut match_img = Mat::default();
                    features2d::draw_matches(
                        &prev_gray,
                        &prev_kp,
                        &gray,
                        &kp,
                        &good_matches,
                        &mut match_img,
                        Scalar::new(0.0, 255.0, 0.0, 0.0),
                        Scalar::new(255.0, 0.0, 0.0, 0.0),
                        &Vector::new(),
                        features2d::DrawMatchesFlags::DEFAULT,
                    )?;
                    highgui::imshow("Matches", &match_img)?;
                }
            }
        }

        if !use_rerun {
            let mut display = frame.clone();
            let info = format!(
                "Frame: {} | Keyframes: {} | 3D Points: {}",
                frame_count,
                keyframe_count,
                global_map.size()
            );
            imgproc::put_text(
                &mut display,
                &info,
                opencv::core::Point::new(10, 30),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.7,
                Scalar::new(0.0, 255.0, 0.0, 0.0),
                2,
                imgproc::LINE_8,
                false,
            )?;
            highgui::imshow("Video", &display)?;

            let traj_img = draw_trajectory(&trajectory, 600, 600)?;
            highgui::imshow("Trajectory", &traj_img)?;

            // Draw 3D map (top-down view)
            let all_points: Vec<MapPoint> =
                global_map.points().iter().map(|p| (*p).clone()).collect();
            let map_img = draw_3d_map(&all_points, &trajectory, 800, 800)?;
            highgui::imshow("3D Map", &map_img)?;

            let key = highgui::wait_key(1)?;
            if key == 'q' as i32 || key == 27 {
                break;
            } else if key == 's' as i32 {
                let all_points: Vec<MapPoint> =
                    global_map.points().iter().map(|p| (*p).clone()).collect();
                save_point_cloud(&all_points)?;
                save_trajectory(&trajectory)?;
            }
        }

        gray.copy_to(&mut prev_gray)?;
        prev_kp = kp;
        prev_desc = desc;
    }

    let elapsed = start_time.elapsed();

    // Save outputs
    let all_points: Vec<MapPoint> = global_map.points().iter().map(|p| (*p).clone()).collect();
    save_point_cloud(&all_points)?;
    save_trajectory(&trajectory)?;

    println!("Total frames: {}", frame_count);
    println!("Keyframes: {}", keyframe_count);
    println!(
        "3D map points: {} ({} stable)",
        global_map.size(),
        global_map.stable_points().len()
    );
    println!("Distance: {:.2}m", trajectory.total_distance());
    println!("Bundle Adjustment runs: {}", ba_runs);
    println!("Time: {:.2}s", elapsed.as_secs_f64());
    println!("Avg FPS: {:.2}", frame_count as f64 / elapsed.as_secs_f64());
    println!("saved: point_cloud.ply, point_cloud.json, trajectory_output.json");

    #[cfg(feature = "rerun")]
    if rec.is_some() {
        println!("Rerun viewer should show the 3D reconstruction!");
        println!("Waiting for data to flush... (press Ctrl+C to exit)");
        std::thread::sleep(std::time::Duration::from_secs(5));
    }

    Ok(())
}

fn transform_points_to_world(
    points: &[MapPoint],
    camera_pose_rt: &(nalgebra::Matrix3<f64>, nalgebra::Vector3<f64>),
) -> Vec<MapPoint> {
    let (r_wtc, t_wtc) = camera_pose_rt; // World-to-Camera

    // We need Camera-to-World to transform points from camera frame to world frame
    // Not sure if we do though, just trying to be thorough
    // subject to change
    let r_ctw = r_wtc.transpose();
    let t_ctw = -r_ctw * t_wtc;

    points
        .iter()
        .map(|p| {
            // Transform point from camera frame to world frame
            let p_world = r_ctw * p.position.coords + t_ctw;
            let mut new_point = p.clone();
            new_point.position = nalgebra::Point3::from(p_world);
            new_point
        })
        .collect()
}

#[cfg(feature = "rerun")]
fn log_frame_to_rerun(
    rec: &rerun::RecordingStream,
    frame: &Mat,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut rgb = Mat::default();
    imgproc::cvt_color(
        frame,
        &mut rgb,
        imgproc::COLOR_BGR2RGB,
        0,
        opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    let width = rgb.cols() as u32;
    let height = rgb.rows() as u32;
    let data = rgb.data_bytes()?.to_vec();

    // Note: No set_time_sequence here, it's done in the main loop
    rec.log(
        "world/camera/image",
        &rerun::Image::from_rgb24(data, [width, height]),
    )?;

    Ok(())
}

#[cfg(feature = "rerun")]
fn log_camera_pose(
    rec: &rerun::RecordingStream,
    pose_rt: &(nalgebra::Matrix3<f64>, nalgebra::Vector3<f64>),
) -> Result<(), Box<dyn std::error::Error>> {
    let (r_wtc, t_wtc) = pose_rt; // World-to-Camera

    //Invert the pose to get Camera-to-World for Rerun
    // as explained subject to change
    let r_ctw = r_wtc.transpose();
    let t_ctw = -r_ctw * t_wtc;

    let translation = glam::Vec3::new(t_ctw[0] as f32, t_ctw[1] as f32, t_ctw[2] as f32);

    let rot = nalgebra::UnitQuaternion::from_rotation_matrix(
        &nalgebra::Rotation3::from_matrix_unchecked(r_ctw),
    );
    let rotation = glam::Quat::from_xyzw(rot.i as f32, rot.j as f32, rot.k as f32, rot.w as f32);

    // Note: No set_time_sequence here, it's done in the main loop
    rec.log(
        "world/camera",
        &rerun::Transform3D::from_translation_rotation(translation, rotation),
    )?;

    Ok(())
}

#[cfg(feature = "rerun")]
fn log_matches_to_rerun(
    rec: &rerun::RecordingStream,
    points1: &Vector<Point2f>,
    points2: &Vector<Point2f>,
) -> Result<(), Box<dyn std::error::Error>> {
    let count = points1.len().min(points2.len()).min(100); // Limit to 100 for visibility

    // Points from the previous keyframe, in green
    let positions1: Vec<[f32; 2]> = (0..count)
        .map(|i| {
            let p = points1.get(i).unwrap();
            [p.x, p.y]
        })
        .collect();

    // Points from the current keyframe, in red
    let positions2: Vec<[f32; 2]> = (0..count)
        .map(|i| {
            let p = points2.get(i).unwrap();
            [p.x, p.y]
        })
        .collect();

    // Log to "world/camera/image" to overlay on the image
    rec.log(
        "world/camera/image",
        &rerun::Points2D::new(positions1)
            .with_colors([[0, 255, 0]]) // Green
            .with_radii([3.0]),
    )?;

    rec.log(
        "world/camera/image",
        &rerun::Points2D::new(positions2)
            .with_colors([[255, 0, 0]]) // Red
            .with_radii([3.0]),
    )?;

    Ok(())
}

#[cfg(feature = "rerun")]
fn log_matches_image_to_rerun(
    rec: &rerun::RecordingStream,
    prev_gray: &Mat,
    gray: &Mat,
    prev_kp: &Vector<KeyPoint>,
    kp: &Vector<KeyPoint>,
    matches: &Vector<opencv::core::DMatch>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut match_img = Mat::default();
    features2d::draw_matches(
        prev_gray,
        prev_kp,
        gray,
        kp,
        matches,
        &mut match_img,
        Scalar::new(0.0, 255.0, 0.0, 0.0), // Green
        Scalar::new(255.0, 0.0, 0.0, 0.0), // Red
        &Vector::new(),
        features2d::DrawMatchesFlags::DEFAULT,
    )?;

    // draw_matches output is BGR, convert to RGB for Rerun
    let mut rgb_match_img = Mat::default();
    imgproc::cvt_color(
        &match_img,
        &mut rgb_match_img,
        imgproc::COLOR_BGR2RGB,
        0,
        opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    let width = rgb_match_img.cols() as u32;
    let height = rgb_match_img.rows() as u32;
    let data = rgb_match_img.data_bytes()?.to_vec();

    rec.log(
        "world/keyframe_matches", // Log to a new, separate entity
        &rerun::Image::from_rgb24(data, [width, height]),
    )?;

    Ok(())
}

#[cfg(feature = "rerun")]
fn log_map_points(
    rec: &rerun::RecordingStream,
    points: &[MapPoint],
) -> Result<(), Box<dyn std::error::Error>> {
    if points.is_empty() {
        return Ok(());
    }

    let positions: Vec<[f32; 3]> = points
        .iter()
        .map(|p| {
            [
                p.position.x as f32,
                p.position.y as f32,
                p.position.z as f32,
            ]
        })
        .collect();

    let colors: Vec<[u8; 3]> = points
        .iter()
        .map(|p| {
            let depth = p.position.z as f32;
            depth_to_color(depth)
        })
        .collect();

    rec.log(
        "world/points",
        &rerun::Points3D::new(positions)
            .with_colors(colors)
            .with_radii([0.02]),
    )?;

    Ok(())
}

#[cfg(feature = "rerun")]
fn log_trajectory(
    rec: &rerun::RecordingStream,
    trajectory: &Trajectory,
) -> Result<(), Box<dyn std::error::Error>> {
    let points = trajectory.points();
    if points.len() < 2 {
        return Ok(());
    }

    let positions: Vec<[f32; 3]> = points
        .iter()
        .map(|p| {
            [
                p.position[0] as f32,
                p.position[1] as f32,
                p.position[2] as f32,
            ]
        })
        .collect();

    rec.log(
        "world/trajectory",
        &rerun::LineStrips3D::new([positions])
            .with_colors([[0, 255, 0]])
            .with_radii([0.01]),
    )?;

    Ok(())
}

#[cfg(feature = "rerun")]
fn depth_to_color(depth: f32) -> [u8; 3] {
    // Simple heatmap: Blue (close) -> Green -> Red (far)
    let normalized = (depth / 50.0).clamp(0.0, 1.0); // Normalize based on expected depth
    if normalized < 0.5 {
        // Blue to Green
        let t = normalized * 2.0; // 0 -> 1
        [0, (255.0 * t) as u8, (255.0 * (1.0 - t)) as u8]
    } else {
        // Green to Red
        let t = (normalized - 0.5) * 2.0; // 0 -> 1
        [(255.0 * t) as u8, (255.0 * (1.0 - t)) as u8, 0]
    }
}

fn draw_trajectory(trajectory: &Trajectory, width: i32, height: i32) -> opencv::Result<Mat> {
    use opencv::core::CV_8UC3;

    let mut img = Mat::new_rows_cols_with_default(
        height,
        width,
        CV_8UC3,
        Scalar::new(255.0, 255.0, 255.0, 0.0),
    )?;

    let points = trajectory.points();
    if points.len() < 2 {
        return Ok(img);
    }

    let mut min_x = f64::MAX;
    let mut max_x = f64::MIN;
    let mut min_z = f64::MAX;
    let mut max_z = f64::MIN;

    for point in points {
        min_x = min_x.min(point.position[0]);
        max_x = max_x.max(point.position[0]);
        min_z = min_z.min(point.position[2]);
        max_z = max_z.max(point.position[2]);
    }

    // Handle case where range is zero to avoid division by zero
    let range_x = (max_x - min_x).max(0.001);
    let range_z = (max_z - min_z).max(0.001);
    let scale = ((width as f64 * 0.8).min(height as f64 * 0.8)) / range_x.max(range_z);

    let offset_x = (width as f64 - range_x * scale) / 2.0;
    let offset_y = (height as f64 + range_z * scale) / 2.0;

    for i in 1..points.len() {
        let p1 = &points[i - 1];
        let p2 = &points[i];

        // Top-down view (X-Z plane)
        let pt1 = opencv::core::Point::new(
            ((p1.position[0] - min_x) * scale + offset_x) as i32,
            (offset_y - (p1.position[2] - min_z) * scale) as i32,
        );

        let pt2 = opencv::core::Point::new(
            ((p2.position[0] - min_x) * scale + offset_x) as i32,
            (offset_y - (p2.position[2] - min_z) * scale) as i32,
        );

        imgproc::line(
            &mut img,
            pt1,
            pt2,
            Scalar::new(255.0, 0.0, 0.0, 0.0),
            2,
            imgproc::LINE_AA,
            0,
        )?;
    }

    // Draw current position as a red circle
    if let Some(p) = points.last() {
        let pt = opencv::core::Point::new(
            ((p.position[0] - min_x) * scale + offset_x) as i32,
            (offset_y - (p.position[2] - min_z) * scale) as i32,
        );
        imgproc::circle(
            &mut img,
            pt,
            5,
            Scalar::new(0.0, 0.0, 255.0, 0.0),
            -1, // filled
            imgproc::LINE_AA,
            0,
        )?;
    }

    Ok(img)
}

fn save_trajectory(trajectory: &Trajectory) -> Result<(), Box<dyn std::error::Error>> {
    let json = trajectory.to_json()?;
    std::fs::write("trajectory_output.json", json)?;
    Ok(())
}

fn save_point_cloud(points: &[MapPoint]) -> Result<(), Box<dyn std::error::Error>> {
    save_ply(points, "point_cloud.ply")?;
    save_json(points, "point_cloud.json")?;
    Ok(())
}

fn save_ply(points: &[MapPoint], filename: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(filename)?;

    writeln!(file, "ply")?;
    writeln!(file, "format ascii 1.0")?;
    writeln!(file, "element vertex {}", points.len())?;
    writeln!(file, "property float x")?;
    writeln!(file, "property float y")?;
    writeln!(file, "property float z")?;
    writeln!(file, "property uchar red")?;
    writeln!(file, "property uchar green")?;
    writeln!(file, "property uchar blue")?;
    writeln!(file, "end_header")?;

    for point in points {
        let depth = point.position.z as f32;
        let color = depth_to_color_ply(depth);
        writeln!(
            file,
            "{} {} {} {} {} {}",
            point.position.x, point.position.y, point.position.z, color.0, color.1, color.2
        )?;
    }

    println!("Saved PLY: {}", filename);
    Ok(())
}

fn save_json(points: &[MapPoint], filename: &str) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(&points)?;
    std::fs::write(filename, json)?;
    println!("Saved JSON: {}", filename);
    Ok(())
}

fn depth_to_color_ply(depth: f32) -> (u8, u8, u8) {
    let normalized = (depth / 50.0).clamp(0.0, 1.0);
    if normalized < 0.5 {
        let t = normalized * 2.0;
        (0, (255.0 * t) as u8, (255.0 * (1.0 - t)) as u8)
    } else {
        let t = (normalized - 0.5) * 2.0;
        ((255.0 * t) as u8, (255.0 * (1.0 - t)) as u8, 0)
    }
}

fn draw_3d_map(
    points: &[MapPoint],
    trajectory: &Trajectory,
    width: i32,
    height: i32,
) -> opencv::Result<Mat> {
    use opencv::core::{CV_8UC3, Scalar};

    let mut img = Mat::new_rows_cols_with_default(
        height,
        width,
        CV_8UC3,
        Scalar::new(20.0, 20.0, 20.0, 0.0),
    )?;

    if points.is_empty() {
        return Ok(img);
    }

    // Find bounds for all points and trajectory
    let mut min_x = f64::MAX;
    let mut max_x = f64::MIN;
    let mut min_z = f64::MAX;
    let mut max_z = f64::MIN;

    for point in points {
        min_x = min_x.min(point.position.x);
        max_x = max_x.max(point.position.x);
        min_z = min_z.min(point.position.z);
        max_z = max_z.max(point.position.z);
    }

    let traj_points = trajectory.points();
    for point in traj_points {
        min_x = min_x.min(point.position[0]);
        max_x = max_x.max(point.position[0]);
        min_z = min_z.min(point.position[2]);
        max_z = max_z.max(point.position[2]);
    }

    let range_x = (max_x - min_x).max(0.001);
    let range_z = (max_z - min_z).max(0.001);
    let scale = ((width as f64 * 0.85).min(height as f64 * 0.85)) / range_x.max(range_z);

    let offset_x = width as f64 * 0.075;
    let offset_y = height as f64 * 0.925;

    for i in 0..10 {
        let x = offset_x + (range_x / 10.0) * i as f64 * scale;
        imgproc::line(
            &mut img,
            opencv::core::Point::new(x as i32, 0),
            opencv::core::Point::new(x as i32, height),
            Scalar::new(40.0, 40.0, 40.0, 0.0),
            1,
            imgproc::LINE_AA,
            0,
        )?;

        let z = offset_y - (range_z / 10.0) * i as f64 * scale;
        imgproc::line(
            &mut img,
            opencv::core::Point::new(0, z as i32),
            opencv::core::Point::new(width, z as i32),
            Scalar::new(40.0, 40.0, 40.0, 0.0),
            1,
            imgproc::LINE_AA,
            0,
        )?;
    }

    for point in points {
        let x = ((point.position.x - min_x) * scale + offset_x) as i32;
        let z = (offset_y - (point.position.z - min_z) * scale) as i32;

        if x >= 0 && x < width && z >= 0 && z < height {
            let color = if point.observations >= 3 {
                Scalar::new(0.0, 255.0, 0.0, 0.0) // Green for stable
            } else {
                Scalar::new(100.0, 100.0, 100.0, 0.0) // Gray for unstable
            };

            imgproc::circle(
                &mut img,
                opencv::core::Point::new(x, z),
                2,
                color,
                -1,
                imgproc::LINE_AA,
                0,
            )?;
        }
    }

    // Draw trajectory
    for i in 1..traj_points.len() {
        let p1 = &traj_points[i - 1];
        let p2 = &traj_points[i];

        let pt1 = opencv::core::Point::new(
            ((p1.position[0] - min_x) * scale + offset_x) as i32,
            (offset_y - (p1.position[2] - min_z) * scale) as i32,
        );

        let pt2 = opencv::core::Point::new(
            ((p2.position[0] - min_x) * scale + offset_x) as i32,
            (offset_y - (p2.position[2] - min_z) * scale) as i32,
        );

        imgproc::line(
            &mut img,
            pt1,
            pt2,
            Scalar::new(255.0, 255.0, 0.0, 0.0),
            2,
            imgproc::LINE_AA,
            0,
        )?;
    }

    // Draw current position
    if let Some(last) = traj_points.last() {
        let pt = opencv::core::Point::new(
            ((last.position[0] - min_x) * scale + offset_x) as i32,
            (offset_y - (last.position[2] - min_z) * scale) as i32,
        );
        imgproc::circle(
            &mut img,
            pt,
            6,
            Scalar::new(0.0, 0.0, 255.0, 0.0),
            -1,
            imgproc::LINE_AA,
            0,
        )?;
    }

    imgproc::put_text(
        &mut img,
        "Green: Stable points | Gray: New points | Yellow: Path | Red: Current",
        opencv::core::Point::new(10, height - 10),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.4,
        Scalar::new(255.0, 255.0, 255.0, 0.0),
        1,
        imgproc::LINE_8,
        false,
    )?;

    Ok(img)
}
