use opencv::{
    core::{Mat, Scalar, Vector},
    features2d, highgui, imgproc,
    prelude::*,
    videoio::{self, VideoCapture, VideoCaptureTrait},
};
use slam_rs::{FeatureMatcher, OrbDetector};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Feature Matching Visualizer\n");

    let video_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "test.mp4".to_string());

    println!("Opening: {}", video_path);
    let mut cap = VideoCapture::from_file(&video_path, videoio::CAP_ANY)?;

    if !cap.is_opened()? {
        return Err("Cannot open video".into());
    }

    let width = cap.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32;
    let height = cap.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32;
    println!("Resolution: {}x{}\n", width, height);

    // Create detector and matcher
    let mut detector = OrbDetector::new(500)?;
    let mut matcher = FeatureMatcher::new()?;

    // Create windows
    highgui::named_window("Matches", highgui::WINDOW_AUTOSIZE)?;

    let mut frame = Mat::default();
    let mut gray = Mat::default();
    let mut prev_gray = Mat::default();
    let mut prev_kp = Vector::new();
    let mut prev_desc = Mat::default();
    let mut frame_count = 0;

    println!("Processing... Press 'q' to quit\n");

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

        let (kp, desc) = detector.detect_and_compute(&gray)?;

        if frame_count > 1 && !prev_desc.empty() {
            // Match features
            let matches = matcher.match_descriptors(&prev_desc, &desc)?;
            let good_matches = matcher.filter_good_matches(&matches, 2.0);

            println!(
                "Frame {}: {} features, {} matches",
                frame_count,
                kp.len(),
                good_matches.len()
            );

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
        }

        // Store for next iteration
        gray.copy_to(&mut prev_gray)?;
        prev_kp = kp;
        prev_desc = desc;

        let key = highgui::wait_key(10)?;
        if key == 'q' as i32 || key == 27 {
            break;
        }
    }

    println!("\nDone! Processed {} frames", frame_count);
    Ok(())
}
