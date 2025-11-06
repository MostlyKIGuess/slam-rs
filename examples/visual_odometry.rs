use opencv::{
    core::{Mat, Point, Scalar, Vector},
    features2d, highgui, imgproc,
    prelude::*,
    videoio::{self, VideoCapture, VideoCaptureTrait},
};
use slamkit_rs::{
    CameraIntrinsics, FeatureMatcher, KeyframeSelector, OrbDetector, PoseEstimator, Trajectory,
};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Visual Odometry Pipeline");

    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!(
            "Usage: {} <video_path> [--fx <fx>] [--fy <fy>] [--cx <cx>] [--cy <cy>]",
            args[0]
        );
        eprintln!("\nExample:");
        eprintln!("  {} video.mp4", args[0]);
        eprintln!(
            "  {} video.mp4 --fx 718.856 --fy 718.856 --cx 607.1928 --cy 185.2157",
            args[0]
        );
        return Err("Missing video path".into());
    }

    let video_path = &args[1];

    // Parse camera intrinsics from command line or use KITTI defaults
    let intrinsics = parse_intrinsics(&args)?;
    println!("Camera Intrinsics:");
    println!("  fx: {:.2}, fy: {:.2}", intrinsics.fx, intrinsics.fy);
    println!("  cx: {:.2}, cy: {:.2}\n", intrinsics.cx, intrinsics.cy);

    println!("Opening video: {}", video_path);
    let mut cap = VideoCapture::from_file(&video_path, videoio::CAP_ANY)?;

    if !cap.is_opened()? {
        return Err("Cannot open video".into());
    }

    let width = cap.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32;
    let height = cap.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32;
    let fps = cap.get(videoio::CAP_PROP_FPS)?;
    let total_frames = cap.get(videoio::CAP_PROP_FRAME_COUNT)? as i32;

    println!("Resolution: {}x{}", width, height);
    println!("FPS: {:.2}", fps);
    if total_frames > 0 {
        println!("Total frames: {}\n", total_frames);
    }

    let mut detector = OrbDetector::new(1000)?;
    let mut matcher = FeatureMatcher::new()?;
    let pose_estimator = PoseEstimator::new(intrinsics);
    let mut trajectory = Trajectory::new();
    let mut keyframe_selector = KeyframeSelector::new();

    highgui::named_window("Video", highgui::WINDOW_AUTOSIZE)?;
    highgui::named_window("Matches", highgui::WINDOW_AUTOSIZE)?;
    highgui::named_window("Trajectory", highgui::WINDOW_AUTOSIZE)?;

    let mut frame = Mat::default();
    let mut gray = Mat::default();
    let mut prev_gray = Mat::default();
    let mut prev_kp = Vector::new();
    let mut prev_desc = Mat::default();
    let mut frame_count = 0;
    let mut successful_frames = 0;
    let mut failed_frames = 0;
    let mut keyframe_count = 0;

    let start_time = Instant::now();
    let mut fps_timer = Instant::now();
    let mut fps_counter = 0;
    let mut processing_fps = 0.0;

    println!("Processing... Press 'q' to quit, 's' to save trajectory\n");

    loop {
        if !cap.read(&mut frame)? || frame.empty() {
            break;
        }

        frame_count += 1;
        fps_counter += 1;

        // Convert to grayscale
        imgproc::cvt_color(
            &frame,
            &mut gray,
            imgproc::COLOR_BGR2GRAY,
            0,
            opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        // Detect features
        let (kp, desc) = detector.detect_and_compute(&gray)?;

        if frame_count > 1 && !prev_desc.empty() {
            // Match features
            let matches = matcher.match_descriptors(&prev_desc, &desc)?;
            let good_matches = matcher.filter_good_matches(&matches, 2.0);

            if good_matches.len() >= 8 {
                // Extract matched points
                let (points1, points2) =
                    pose_estimator.extract_matched_points(&prev_kp, &kp, &good_matches)?;

                // Estimate pose
                match pose_estimator.compute_essential_matrix(&points1, &points2) {
                    Ok(essential) => {
                        match pose_estimator.recover_pose(&essential, &points1, &points2) {
                            Ok((rotation, translation)) => {
                                // Check if this should be a keyframe
                                let is_keyframe = keyframe_selector.should_be_keyframe(
                                    &rotation,
                                    &translation,
                                    good_matches.len(),
                                );

                                if is_keyframe {
                                    keyframe_count += 1;
                                    // Update trajectory only on keyframes
                                    let timestamp = (frame_count - 1) as f64 / fps;
                                    trajectory.update(
                                        &rotation,
                                        &translation,
                                        frame_count,
                                        timestamp,
                                    );
                                }

                                successful_frames += 1;

                                // Print progress
                                if frame_count % 30 == 0 {
                                    println!(
                                        "Frame {:4} | Matches: {:3} | Keyframes: {:3} | Distance: {:.2}m",
                                        frame_count,
                                        good_matches.len(),
                                        keyframe_count,
                                        trajectory.total_distance(),
                                    );
                                }
                            }
                            Err(_) => {
                                failed_frames += 1;
                            }
                        }
                    }
                    Err(_) => {
                        failed_frames += 1;
                    }
                }

                // Draw matches
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
            } else {
                failed_frames += 1;
            }
        }

        // Calculate FPS
        if fps_timer.elapsed().as_secs() >= 1 {
            processing_fps = fps_counter as f64 / fps_timer.elapsed().as_secs_f64();
            fps_counter = 0;
            fps_timer = Instant::now();
        }

        // Draw info overlay on video
        let mut display = frame.clone();
        let info = format!("Frame: {} | FPS: {:.1}", frame_count, processing_fps);
        draw_text(&mut display, &info, Point::new(10, 30))?;

        let traj_info = format!(
            "Keyframes: {} | Distance: {:.2}m",
            keyframe_count,
            trajectory.total_distance()
        );
        draw_text(&mut display, &traj_info, Point::new(10, 60))?;

        highgui::imshow("Video", &display)?;

        // Visualize trajectory
        let traj_img = draw_trajectory(&trajectory, 600, 600)?;
        highgui::imshow("Trajectory", &traj_img)?;

        // Store for next iteration
        gray.copy_to(&mut prev_gray)?;
        prev_kp = kp;
        prev_desc = desc;

        // Handle keyboard input
        let key = highgui::wait_key(1)?;
        if key == 'q' as i32 || key == 27 {
            break;
        } else if key == 's' as i32 {
            save_trajectory(&trajectory)?;
        }
    }

    let elapsed = start_time.elapsed();

    save_trajectory(&trajectory)?;

    println!("\nSummary ");
    println!("Total frames: {}", frame_count);
    println!("Successful poses: {}", successful_frames);
    println!("Failed poses: {}", failed_frames);
    println!("Keyframes selected: {}", keyframe_count);
    println!(
        "Keyframe ratio: {:.1}%",
        (keyframe_count as f64 / frame_count as f64) * 100.0
    );
    println!("Total distance: {:.2}m", trajectory.total_distance());
    println!("Total time: {:.2}s", elapsed.as_secs_f64());
    println!(
        "Average FPS: {:.2}",
        frame_count as f64 / elapsed.as_secs_f64()
    );
    println!("\nTrajectory saved to: trajectory_output.json");

    Ok(())
}

fn parse_intrinsics(args: &[String]) -> Result<CameraIntrinsics, Box<dyn std::error::Error>> {
    let mut fx = None;
    let mut fy = None;
    let mut cx = None;
    let mut cy = None;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--fx" => {
                fx = Some(args[i + 1].parse()?);
                i += 2;
            }
            "--fy" => {
                fy = Some(args[i + 1].parse()?);
                i += 2;
            }
            "--cx" => {
                cx = Some(args[i + 1].parse()?);
                i += 2;
            }
            "--cy" => {
                cy = Some(args[i + 1].parse()?);
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }

    // If all provided, use custom; otherwise use KITTI defaults
    if let (Some(fx), Some(fy), Some(cx), Some(cy)) = (fx, fy, cx, cy) {
        println!("Using custom camera intrinsics\n");
        Ok(CameraIntrinsics::new(fx, fy, cx, cy))
    } else {
        println!("Using KITTI default intrinsics\n");
        Ok(CameraIntrinsics::kitti())
    }
}

fn draw_text(img: &mut Mat, text: &str, pos: Point) -> Result<(), Box<dyn std::error::Error>> {
    imgproc::put_text(
        img,
        text,
        pos,
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.6,
        Scalar::new(0.0, 255.0, 0.0, 0.0),
        2,
        imgproc::LINE_8,
        false,
    )?;
    Ok(())
}

fn draw_trajectory(
    trajectory: &Trajectory,
    width: i32,
    height: i32,
) -> Result<Mat, Box<dyn std::error::Error>> {
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

    // Find bounds for scaling
    let mut min_x = f64::MAX;
    let mut max_x = f64::MIN;
    let mut min_z = f64::MAX;
    let mut max_z = f64::MIN;

    for pt in points {
        let x = pt.position[0];
        let z = pt.position[2];
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_z = min_z.min(z);
        max_z = max_z.max(z);
    }

    let range_x = (max_x - min_x).max(1.0);
    let range_z = (max_z - min_z).max(1.0);
    let scale = ((width as f64 - 40.0) / range_x).min((height as f64 - 40.0) / range_z);

    // Draw trajectory
    for i in 1..points.len() {
        let p1 = &points[i - 1];
        let p2 = &points[i];

        let pt1 = Point::new(
            ((p1.position[0] - min_x) * scale + 20.0) as i32,
            (height as f64 - (p1.position[2] - min_z) * scale - 20.0) as i32,
        );
        let pt2 = Point::new(
            ((p2.position[0] - min_x) * scale + 20.0) as i32,
            (height as f64 - (p2.position[2] - min_z) * scale - 20.0) as i32,
        );

        // Color from blue (start) to red (end)
        let ratio = i as f64 / points.len() as f64;
        let color = Scalar::new(255.0 * (1.0 - ratio), 0.0, 255.0 * ratio, 0.0);

        imgproc::line(&mut img, pt1, pt2, color, 2, imgproc::LINE_AA, 0)?;
    }

    // Draw start point (green)
    let start = &points[0];
    let start_pt = Point::new(
        ((start.position[0] - min_x) * scale + 20.0) as i32,
        (height as f64 - (start.position[2] - min_z) * scale - 20.0) as i32,
    );
    imgproc::circle(
        &mut img,
        start_pt,
        5,
        Scalar::new(0.0, 255.0, 0.0, 0.0),
        -1,
        imgproc::LINE_AA,
        0,
    )?;

    // Draw end point (red)
    let end = points.last().unwrap();
    let end_pt = Point::new(
        ((end.position[0] - min_x) * scale + 20.0) as i32,
        (height as f64 - (end.position[2] - min_z) * scale - 20.0) as i32,
    );
    imgproc::circle(
        &mut img,
        end_pt,
        5,
        Scalar::new(0.0, 0.0, 255.0, 0.0),
        -1,
        imgproc::LINE_AA,
        0,
    )?;

    Ok(img)
}

fn save_trajectory(trajectory: &Trajectory) -> Result<(), Box<dyn std::error::Error>> {
    trajectory.save_to_file("trajectory_output.json")?;
    println!("Trajectory saved!");
    Ok(())
}
