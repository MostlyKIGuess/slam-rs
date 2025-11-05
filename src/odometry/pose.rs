use nalgebra as na;
use opencv::{
    calib3d::{self, RANSAC},
    core::{Mat, Point2f, Vector},
    prelude::*,
};

use super::camera::CameraIntrinsics;

/// Pose estimator for VO
pub struct PoseEstimator {
    intrinsics: CameraIntrinsics,
    min_matches: usize,
}

impl PoseEstimator {
    /// Create a new pose estimator with the given camera intrinsics.
    pub fn new(intrinsics: CameraIntrinsics) -> Self {
        Self {
            intrinsics,
            min_matches: 8,
        }
    }

    /// Extract Point2f coordinates from matched keypoints
    /// This is basically the process of converting matched keypoints into a format suitable for pose estimation.
    pub fn extract_matched_points(
        &self,
        kp1: &Vector<opencv::core::KeyPoint>,
        kp2: &Vector<opencv::core::KeyPoint>,
        matches: &Vector<opencv::core::DMatch>,
    ) -> Result<(Vector<Point2f>, Vector<Point2f>), Box<dyn std::error::Error>> {
        let mut points1 = Vector::new();
        let mut points2 = Vector::new();

        for m in matches.iter() {
            let pt1 = kp1.get(m.query_idx as usize)?.pt();
            let pt2 = kp2.get(m.train_idx as usize)?.pt();
            points1.push(pt1);
            points2.push(pt2);
        }

        Ok((points1, points2))
    }

    /// Compute essential matrix from matched points
    pub fn compute_essential_matrix(
        &self,
        points1: &Vector<Point2f>,
        points2: &Vector<Point2f>,
    ) -> Result<Mat, Box<dyn std::error::Error>> {
        if points1.len() < self.min_matches || points2.len() < self.min_matches {
            return Err(format!(
                "Insufficient points: {} (need {})",
                points1.len(),
                self.min_matches
            )
            .into());
        }

        let camera_matrix = self.intrinsics.to_matrix()?;
        let mut mask = Mat::default();

        let essential = calib3d::find_essential_mat(
            points1,
            points2,
            &camera_matrix,
            RANSAC,
            0.999, // confidence , these are c++ default values
            1.0,   // threshold
            1000,  // max_iters
            &mut mask,
        )?;

        if essential.empty() {
            return Err("Failed to compute essential matrix".into());
        }

        Ok(essential)
    }

    /// Recover rotation and translation from essential matrix
    pub fn recover_pose(
        &self,
        essential: &Mat,
        points1: &Vector<Point2f>,
        points2: &Vector<Point2f>,
    ) -> Result<(na::Matrix3<f64>, na::Vector3<f64>), Box<dyn std::error::Error>> {
        let camera_matrix = self.intrinsics.to_matrix()?;
        let mut r = Mat::default();
        let mut t = Mat::default();
        let mut mask = Mat::default();

        let inliers = calib3d::recover_pose_estimated(
            essential,
            points1,
            points2,
            &camera_matrix,
            &mut r,
            &mut t,
            // 1.0, // focal length (unused when camera matrix provided)
            // opencv::core::Point2d::new(0.0, 0.0), // principal point (unused)
            &mut mask,
        )?;

        if inliers < self.min_matches as i32 {
            return Err(format!("Too few inliers: {}", inliers).into());
        }

        let rotation = mat_to_rotation3(&r)?;
        let translation = mat_to_vector3(&t)?;

        Ok((rotation, translation))
    }
}

/// Convert OpenCV 3x3 Mat to nalgebra Matrix3
fn mat_to_rotation3(mat: &Mat) -> Result<na::Matrix3<f64>, Box<dyn std::error::Error>> {
    if mat.rows() != 3 || mat.cols() != 3 {
        return Err("Invalid rotation matrix dimensions".into());
    }

    let mut rotation = na::Matrix3::zeros();
    for i in 0..3 {
        for j in 0..3 {
            rotation[(i, j)] = *mat.at_2d::<f64>(i as i32, j as i32)?;
        }
    }

    Ok(rotation)
}

/// Convert OpenCV 3x1 Mat to nalgebra Vector3
fn mat_to_vector3(mat: &Mat) -> Result<na::Vector3<f64>, Box<dyn std::error::Error>> {
    if mat.rows() != 3 || mat.cols() != 1 {
        return Err("Invalid translation vector dimensions".into());
    }

    let x = *mat.at_2d::<f64>(0, 0)?;
    let y = *mat.at_2d::<f64>(1, 0)?;
    let z = *mat.at_2d::<f64>(2, 0)?;

    Ok(na::Vector3::new(x, y, z))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pose_estimator_creation() {
        let cam = CameraIntrinsics::kitti();
        let estimator = PoseEstimator::new(cam);
        assert_eq!(estimator.min_matches, 8);
    }

    #[test]
    fn test_insufficient_points() {
        let cam = CameraIntrinsics::kitti();
        let estimator = PoseEstimator::new(cam);

        let mut points = Vector::new();
        for i in 0..5 {
            points.push(Point2f::new(i as f32, i as f32));
        }

        let result = estimator.compute_essential_matrix(&points, &points);
        assert!(result.is_err());
    }

    #[test]
    fn test_mat_conversion() -> opencv::Result<()> {
        use opencv::core::CV_64F;
        let mut mat =
            Mat::new_rows_cols_with_default(3, 3, CV_64F, opencv::core::Scalar::all(0.0))?;
        *mat.at_2d_mut::<f64>(0, 0)? = 1.0;
        *mat.at_2d_mut::<f64>(1, 1)? = 1.0;
        *mat.at_2d_mut::<f64>(2, 2)? = 1.0;

        // TODO: we should think of handling (&mat)? and let mat_to_rotation3 return opencv::Result,
        // This is for later TODO
        let rotation = mat_to_rotation3(&mat)
            .map_err(|e| opencv::Error::new(opencv::core::StsError, e.to_string()))?;
        assert_eq!(rotation, na::Matrix3::identity());
        Ok(())
    }
}
