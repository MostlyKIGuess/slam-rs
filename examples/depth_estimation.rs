use clap::Parser;
use opencv::{
    core::{Mat, Vector},
    highgui, imgcodecs, imgproc,
    prelude::*,
    videoio::{self, VideoCapture, VideoCaptureTrait},
};

const MAGMA_COLORMAP_PATH: &str = "src/depth/magma.png";
use std::time::Instant;

#[cfg(feature = "depth")]
use slamkit_rs::MonoDepth2;

#[cfg(feature = "rerun")]
use rerun::RecordingStreamBuilder;

#[derive(Parser, Debug)]
#[command(version, about = "Monocular depth estimation example")]
struct Cli {
    #[arg(help = "Path to input video or image")]
    input: String,

    #[arg(long, default_value = "weights/encoder.pt")]
    encoder: String,

    #[arg(long, default_value = "weights/depth.pt")]
    decoder: String,

    #[arg(long, help = "Use CUDA GPU acceleration")]
    cuda: bool,

    #[arg(long, default_value = "640")]
    width: i32,

    #[arg(long, default_value = "192")]
    height: i32,

    #[arg(long, help = "Use Rerun for visualization")]
    rerun: bool,

    #[arg(long, help = "Save depth maps to disk")]
    save: bool,

    #[arg(long, default_value = "1", help = "Process every Nth frame")]
    skip_frames: usize,
}

#[cfg(not(feature = "depth"))]
fn main() {
    eprintln!("Error: Depth estimation not enabled!");
    eprintln!(
        "Please compile with: cargo run --example depth_estimation --features depth -- <args>"
    );
    std::process::exit(1);
}

#[cfg(feature = "depth")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    println!("Depth Estimation with MonoDepth2");

    // Check if input is video or image
    let is_video =
        cli.input.ends_with(".mp4") || cli.input.ends_with(".avi") || cli.input.ends_with(".mov");

    // Initialize MonoDepth2
    println!("Loading MonoDepth2 model...");
    println!("  Encoder: {}", cli.encoder);
    println!("  Decoder: {}", cli.decoder);
    println!("  Device: {}", if cli.cuda { "CUDA" } else { "CPU" });
    println!("  Resolution: {}x{}\n", cli.width, cli.height);

    let depth_model =
        match MonoDepth2::new(&cli.encoder, &cli.decoder, cli.cuda, cli.width, cli.height) {
            Ok(model) => {
                println!("Model loaded successfully!");
                if model.is_cuda() {
                    println!("Using GPU acceleration");
                } else {
                    println!("Using CPU");
                }
                model
            }
            Err(e) => {
                eprintln!("Failed to load model: {}", e);
                return Err(e);
            }
        };

    println!();

    #[cfg(feature = "rerun")]
    let rec = if cli.rerun {
        println!("Starting Rerun viewer...");
        Some(RecordingStreamBuilder::new("depth-estimation").spawn()?)
    } else {
        None
    };

    // Define `rec` as None when the rerun feature is not enabled
    #[cfg(not(feature = "rerun"))]
    let rec: Option<()> = {
        if cli.rerun {
            eprintln!("Warning: Rerun not available. Build with --features depth,rerun");
        }
        None
    };

    if !cli.rerun {
        highgui::named_window("Input", highgui::WINDOW_AUTOSIZE)?;
        highgui::named_window("Depth", highgui::WINDOW_AUTOSIZE)?;
    }

    if is_video {
        process_video(&depth_model, &cli, rec.as_ref())?;
    } else {
        process_image(&depth_model, &cli, rec.as_ref())?;
    }

    println!("\nDone!");

    #[cfg(feature = "rerun")]
    if rec.is_some() {
        println!("Rerun viewer should be open. Press Ctrl+C to exit.");
        std::thread::sleep(std::time::Duration::from_secs(5));
    }

    Ok(())
}

#[cfg(feature = "depth")]
fn process_image(
    depth_model: &MonoDepth2,
    cli: &Cli,
    // compile the type of `rec`
    #[cfg(feature = "rerun")]
    #[allow(unused_variables)]
    rec: Option<&rerun::RecordingStream>,
    #[cfg(not(feature = "rerun"))]
    #[allow(unused_variables)]
    rec: Option<&()>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Processing image: {}", cli.input);

    let image = imgcodecs::imread(&cli.input, imgcodecs::IMREAD_COLOR)?;
    if image.empty() {
        return Err(format!("Failed to load image: {}", cli.input).into());
    }

    println!("Image size: {}x{}", image.cols(), image.rows());

    let start = Instant::now();
    let depth_colored = depth_model.predict_colored(&image, MAGMA_COLORMAP_PATH)?;
    let duration = start.elapsed();

    println!(
        "Depth prediction time: {:.2}ms",
        duration.as_secs_f64() * 1000.0
    );

    if cli.save {
        let output_path = "depth_output.jpg";
        imgcodecs::imwrite(output_path, &depth_colored, &Vector::new())?;
        println!("Saved: {}", output_path);
    }

    #[cfg(feature = "rerun")]
    if let Some(rec) = rec {
        log_to_rerun(rec, &image, &depth_colored, 0)?;
        println!("Logged to Rerun viewer");
    }

    if !cli.rerun {
        highgui::imshow("Input", &image)?;
        highgui::imshow("Depth", &depth_colored)?;
        println!("Press any key to exit...");
        highgui::wait_key(0)?;
    }

    Ok(())
}

#[cfg(feature = "depth")]
fn process_video(
    depth_model: &MonoDepth2,
    cli: &Cli,
    #[cfg(feature = "rerun")]
    #[allow(unused_variables)]
    rec: Option<&rerun::RecordingStream>,
    #[cfg(not(feature = "rerun"))]
    #[allow(unused_variables)]
    rec: Option<&()>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Opening video: {}", cli.input);

    let mut cap = VideoCapture::from_file(&cli.input, videoio::CAP_ANY)?;
    if !cap.is_opened()? {
        return Err(format!("Failed to open video: {}", cli.input).into());
    }

    let width = cap.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32;
    let height = cap.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32;
    let fps = cap.get(videoio::CAP_PROP_FPS)?;
    let total_frames = cap.get(videoio::CAP_PROP_FRAME_COUNT)? as usize;

    println!("Video info:");
    println!("  Resolution: {}x{}", width, height);
    println!("  FPS: {:.2}", fps);
    println!("  Total frames: {}", total_frames);
    println!("  Processing every {} frame(s)\n", cli.skip_frames);

    let mut frame = Mat::default();
    let mut frame_count = 0;
    let mut processed_count = 0;
    let start_time = Instant::now();
    let mut total_inference_time = 0.0;

    println!("Processing... Press 'q' to quit, 's' to save current frame");

    loop {
        if !cap.read(&mut frame)? || frame.empty() {
            break;
        }

        frame_count += 1;

        if frame_count % cli.skip_frames != 0 {
            continue;
        }

        processed_count += 1;

        let inference_start = Instant::now();
        let depth_colored = depth_model.predict_colored(&frame, MAGMA_COLORMAP_PATH)?;
        let inference_time = inference_start.elapsed().as_secs_f64() * 1000.0;
        total_inference_time += inference_time;

        // `rerun` is disabled
        #[cfg(feature = "rerun")]
        if let Some(rec) = rec {
            rec.set_time_sequence("frame", frame_count as i64);
            log_to_rerun(rec, &frame, &depth_colored, frame_count)?;
        }

        if !cli.rerun {
            let mut display = frame.clone();
            let info = format!(
                "Frame: {}/{} | Inference: {:.1}ms | Avg: {:.1}ms",
                frame_count,
                total_frames,
                inference_time,
                total_inference_time / processed_count as f64
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

            highgui::imshow("Input", &display)?;
            highgui::imshow("Depth", &depth_colored)?;

            let key = highgui::wait_key(1)?;
            if key == 'q' as i32 || key == 27 {
                break;
            } else if key == 's' as i32 {
                let filename = format!("depth_frame_{:06}.jpg", frame_count);
                imgcodecs::imwrite(&filename, &depth_colored, &Vector::new())?;
                println!("Saved: {}", filename);
            }
        }

        if processed_count % 30 == 0 {
            let avg_inference = total_inference_time / processed_count as f64;
            let avg_fps = processed_count as f64 / start_time.elapsed().as_secs_f64();
            println!(
                "Frame {}/{} | Avg inference: {:.1}ms | Avg FPS: {:.1}",
                frame_count, total_frames, avg_inference, avg_fps
            );
        }
    }

    let elapsed = start_time.elapsed();
    let avg_inference = total_inference_time / processed_count as f64;
    let avg_fps = processed_count as f64 / elapsed.as_secs_f64();

    println!("\nStatistics:");
    println!("  Total frames: {}", frame_count);
    println!("  Processed frames: {}", processed_count);
    println!("  Average inference time: {:.1}ms", avg_inference);
    println!("  Average FPS: {:.1}", avg_fps);
    println!("  Total time: {:.2}s", elapsed.as_secs_f64());

    Ok(())
}

#[cfg(all(feature = "depth", feature = "rerun"))]
fn log_to_rerun(
    rec: &rerun::RecordingStream,
    rgb_image: &Mat,
    depth_image: &Mat,
    frame_idx: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    use rerun::TensorData;
    use rerun::datatypes::TensorBuffer;
    use rerun::{ColorModel, Image};

    let rgb_data = rgb_image.data_bytes()?.to_vec();
    let (w, h) = (rgb_image.cols() as u32, rgb_image.rows() as u32);

    let rgb_tensor = TensorData::new(
        vec![h as u64, w as u64, 3],
        TensorBuffer::U8(rgb_data.into()),
    );
    let rgb_image = Image::from_color_model_and_tensor(ColorModel::BGR, rgb_tensor)?;
    rec.log("camera/rgb", &rgb_image)?;

    let depth_data = depth_image.data_bytes()?.to_vec();
    let (wd, hd) = (depth_image.cols() as u32, depth_image.rows() as u32);

    let depth_tensor = TensorData::new(
        vec![hd as u64, wd as u64, 3],
        TensorBuffer::U8(depth_data.into()),
    );
    let depth_image = Image::from_color_model_and_tensor(ColorModel::BGR, depth_tensor)?;
    rec.log("camera/depth_colored", &depth_image)?;

    rec.log(
        "info",
        &rerun::TextDocument::new(format!("Frame: {}", frame_idx)),
    )?;

    Ok(())
}
