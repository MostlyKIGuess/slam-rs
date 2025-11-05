use opencv::{
    core::Mat,
    features2d::ORB,
    imgproc,
    prelude::*,
    videoio::{self, VideoCapture, VideoCaptureTrait},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("SLAM-RS: Starting...\n");

    // Get video path from command line
    let video_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "test.mp4".to_string());

    println!("Opening video: {}", video_path);
    let mut cap = VideoCapture::from_file(&video_path, videoio::CAP_ANY)?;

    if !cap.is_opened()? {
        return Err("Cannot open video".into());
    }

    // Get video info
    let width = cap.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32;
    let height = cap.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32;
    println!("Resolution: {}x{}\n", width, height);

    // Create ORB detector
    let mut orb = ORB::create_def()?;

    let mut frame = Mat::default();
    let mut gray = Mat::default();
    let mut frame_count = 0;

    println!("Processing frames...");
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

        let mut keypoints = opencv::core::Vector::new();
        orb.detect(&gray, &mut keypoints, &Mat::default())?;

        if frame_count % 30 == 0 {
            println!(
                "Frame {}: {} features detected",
                frame_count,
                keypoints.len()
            );
        }
    }

    println!("Done! Processed {} frames", frame_count);
    Ok(())
}
