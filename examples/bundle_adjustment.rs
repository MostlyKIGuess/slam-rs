use nalgebra as na;
use slamkit_rs::{BundleAdjuster, CameraIntrinsics, Observation};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Bundle Adjustment Example");

    let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0);
    println!("Camera intrinsics:");
    println!("  fx: {}, fy: {}", intrinsics.fx, intrinsics.fy);
    println!("  cx: {}, cy: {}\n", intrinsics.cx, intrinsics.cy);

    let true_points = vec![
        na::Point3::new(0.0, 0.0, 10.0),
        na::Point3::new(2.0, 1.0, 10.0),
        na::Point3::new(-1.0, 2.0, 12.0),
        na::Point3::new(1.5, -1.5, 11.0),
    ];

    let true_poses = vec![
        (na::Matrix3::identity(), na::Vector3::zeros()),
        (na::Matrix3::identity(), na::Vector3::new(1.0, 0.0, 0.0)),
        (na::Matrix3::identity(), na::Vector3::new(2.0, 0.5, 0.0)),
        (na::Matrix3::identity(), na::Vector3::new(3.0, 1.0, 0.0)),
    ];

    println!("Ground truth setup:");
    println!("  {} 3D points", true_points.len());
    println!("  {} camera poses\n", true_poses.len());

    let mut observations = Vec::new();
    for (point_idx, true_point) in true_points.iter().enumerate() {
        for (pose_idx, (r, t)) in true_poses.iter().enumerate() {
            let p_cam = r * true_point + t;
            if p_cam.z > 0.0 {
                let x = intrinsics.fx * (p_cam.x / p_cam.z) + intrinsics.cx;
                let y = intrinsics.fy * (p_cam.y / p_cam.z) + intrinsics.cy;
                observations.push(Observation::new(pose_idx, point_idx, na::Point2::new(x, y)));
            }
        }
    }

    println!("Generated {} observations\n", observations.len());

    let mut noisy_poses = true_poses.clone();
    for pose in &mut noisy_poses {
        pose.1.x += 0.1 * (rand::random::<f64>() - 0.5);
        pose.1.y += 0.1 * (rand::random::<f64>() - 0.5);
        pose.1.z += 0.1 * (rand::random::<f64>() - 0.5);
    }

    let mut noisy_points: Vec<na::Point3<f64>> = true_points
        .iter()
        .map(|p| {
            na::Point3::new(
                p.x + 0.2 * (rand::random::<f64>() - 0.5),
                p.y + 0.2 * (rand::random::<f64>() - 0.5),
                p.z + 0.3 * (rand::random::<f64>() - 0.5),
            )
        })
        .collect();

    let ba = BundleAdjuster::new(intrinsics)
        .with_max_iterations(30)
        .with_lambda(1e-3);

    let initial_reproj_error = ba.compute_total_error(&noisy_poses, &noisy_points, &observations);

    println!("Initial errors (before optimization):");
    println!("  Reprojection error: {:.4}", initial_reproj_error);
    for (i, (noisy, true_pt)) in noisy_points.iter().zip(true_points.iter()).enumerate() {
        let error = (noisy.coords - true_pt.coords).norm();
        println!("  Point {} distance to truth: {:.4}", i, error);
    }

    for (i, (noisy, true_pose)) in noisy_poses.iter().zip(true_poses.iter()).enumerate() {
        let t_error = (noisy.1 - true_pose.1).norm();
        println!("  Pose {} translation error: {:.4}", i, t_error);
    }
    println!();

    println!("Running bundle adjustment...");
    println!("  Max iterations: 30");
    println!("  Lambda: 1e-3\n");

    let start = std::time::Instant::now();
    let final_error = ba.optimize(&mut noisy_poses, &mut noisy_points, &observations, true)?;
    let duration = start.elapsed();

    println!(
        "Optimization completed in {:.2}ms",
        duration.as_secs_f64() * 1000.0
    );
    println!(
        "Reprojection error: {:.4} -> {:.4}",
        initial_reproj_error, final_error
    );
    println!(
        "Improvement: {:.2}%\n",
        (initial_reproj_error - final_error) / initial_reproj_error * 100.0
    );

    println!("Distance to ground truth (after optimization):");
    println!("Note: BA optimizes reprojection error, not distance to truth!");
    for (i, (refined, true_pt)) in noisy_points.iter().zip(true_points.iter()).enumerate() {
        let error = (refined.coords - true_pt.coords).norm();
        println!("  Point {} distance: {:.4}", i, error);
    }

    for (i, (refined, true_pose)) in noisy_poses.iter().zip(true_poses.iter()).enumerate() {
        let t_error = (refined.1 - true_pose.1).norm();
        println!("  Pose {} translation error: {:.4}", i, t_error);
    }
    println!("Bundle adjustment successfully minimized reprojection error!");

    println!("Local Bundle Adjustment Demo ");

    let mut test_poses = true_poses.clone();
    for pose in &mut test_poses {
        pose.1.x += 0.1 * (rand::random::<f64>() - 0.5);
        pose.1.y += 0.1 * (rand::random::<f64>() - 0.5);
        pose.1.z += 0.1 * (rand::random::<f64>() - 0.5);
    }

    let mut test_points = true_points.clone();
    for point in &mut test_points {
        point.x += 0.2 * (rand::random::<f64>() - 0.5);
        point.y += 0.2 * (rand::random::<f64>() - 0.5);
        point.z += 0.3 * (rand::random::<f64>() - 0.5);
    }

    println!("Running local BA with window size 2...");
    let window_size = 2;
    let local_error = ba.local_bundle_adjustment(
        &mut test_poses,
        &mut test_points,
        &observations,
        window_size,
    )?;

    println!("Local BA final error: {:.4}", local_error);
    println!("Only optimized last {} keyframes", window_size);

    Ok(())
}
