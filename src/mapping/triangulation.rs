use nalgebra as na;
use opencv::{
    calib3d,
    core::{CV_64F, Mat, Point2f, Vector},
    prelude::*,
};
use serde::{Deserialize, Serialize};

use crate::odometry::CameraIntrinsics;

/// A 3D map point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MapPoint {
    /// 3D position in world coordinates
    pub position: na::Point3<f64>,
    /// Descriptor (for matching)
    pub descriptor: Option<Vec<u8>>,
    /// Number of times this point has been observed
    pub observations: usize,
    /// ID of the point
    pub id: usize,
}

impl MapPoint {
    /// Create a new map point
    pub fn new(position: na::Point3<f64>, id: usize) -> Self {
        Self {
            position,
            descriptor: None,
            observations: 1,
            id,
        }
    }

    /// Create a new map point with descriptor
    pub fn with_descriptor(position: na::Point3<f64>, descriptor: Vec<u8>, id: usize) -> Self {
        Self {
            position,
            descriptor: Some(descriptor),
            observations: 1,
            id,
        }
    }

    /// Add observation count
    pub fn add_observation(&mut self) {
        self.observations += 1;
    }
}

/// Triangulator for computing 3D points from 2D correspondences
pub struct Triangulator {
    intrinsics: CameraIntrinsics,
    /// Minimum parallax angle in degrees (to ensure good triangulation)
    min_parallax_deg: f64,
    /// Maximum reprojection error to accept a point
    max_reproj_error: f64,
}

impl Triangulator {
    /// Create a new triangulator with the given camera intrinsics
    pub fn new(intrinsics: CameraIntrinsics) -> Self {
        Self {
            intrinsics,
            min_parallax_deg: 1.0,
            max_reproj_error: 4.0, // TODO: Implement a better default value
        }
    }

    /// Set minimum parallax angle in degrees
    pub fn with_min_parallax(mut self, deg: f64) -> Self {
        self.min_parallax_deg = deg;
        self
    }

    /// Set maximum reprojection error
    pub fn with_max_reproj_error(mut self, error: f64) -> Self {
        self.max_reproj_error = error;
        self
    }

    /// Triangulate 3D points from two camera poses and matched 2D points
    ///
    /// Arguments
    /// - `pose1` - (R, t) of the first camera (world to camera transform)
    /// - `pose2` - (R, t) of the second camera (world to camera transform)
    /// - `points1` - 2D points in the first image
    /// - `points2` - 2D points in the second image
    ///
    /// Returns
    /// Vector of successfully triangulated MapPoints
    pub fn triangulate(
        &self,
        pose1: &(na::Matrix3<f64>, na::Vector3<f64>),
        pose2: &(na::Matrix3<f64>, na::Vector3<f64>),
        points1: &Vector<Point2f>,
        points2: &Vector<Point2f>,
    ) -> Result<Vec<MapPoint>, Box<dyn std::error::Error>> {
        if points1.len() != points2.len() {
            return Err("Point arrays must have the same length".into());
        }

        if points1.is_empty() {
            return Ok(Vec::new());
        }

        // Build projection matrices P1 = K * [R1 | t1] and P2 = K * [R2 | t2]
        let proj1 = self.build_projection_matrix(&pose1.0, &pose1.1)?;
        let proj2 = self.build_projection_matrix(&pose2.0, &pose2.1)?;

        // Triangulate points using OpenCV
        let mut points_4d = Mat::default();
        calib3d::triangulate_points(&proj1, &proj2, points1, points2, &mut points_4d)?;

        // Convert homogeneous 4D points to 3D and filter
        // Note: triangulatePoints outputs CV_32F, so we read as f32 and convert to f64
        let mut map_points = Vec::new();
        for i in 0..points_4d.cols() {
            let x = *points_4d.at_2d::<f32>(0, i)? as f64;
            let y = *points_4d.at_2d::<f32>(1, i)? as f64;
            let z = *points_4d.at_2d::<f32>(2, i)? as f64;
            let w = *points_4d.at_2d::<f32>(3, i)? as f64;

            // Convert from homogeneous coordinates
            if w.abs() < 1e-10 {
                continue; // Skip points at infinity
            }

            let point_3d = na::Point3::new(x / w, y / w, z / w);

            // Check if point is in front of both cameras
            if !self.is_in_front_of_camera(&point_3d, &pose1.0, &pose1.1) {
                continue;
            }
            if !self.is_in_front_of_camera(&point_3d, &pose2.0, &pose2.1) {
                continue;
            }

            // TODO: Check parallax angle
            // (For now, skipping this check for simplicity)

            map_points.push(MapPoint::new(point_3d, i as usize));
        }

        Ok(map_points)
    }

    /// Build projection matrix P = K * [R | t]
    fn build_projection_matrix(
        &self,
        r: &na::Matrix3<f64>,
        t: &na::Vector3<f64>,
    ) -> Result<Mat, Box<dyn std::error::Error>> {
        let k = self.intrinsics.to_matrix()?;

        // Create [R | t] as a 3x4 matrix
        let mut rt = Mat::new_rows_cols_with_default(3, 4, CV_64F, opencv::core::Scalar::all(0.0))?;

        for i in 0..3 {
            for j in 0..3 {
                *rt.at_2d_mut::<f64>(i as i32, j as i32)? = r[(i, j)];
            }
            *rt.at_2d_mut::<f64>(i as i32, 3)? = t[i];
        }

        // Compute P = K * [R | t]
        let mut proj = Mat::default();
        opencv::core::gemm(&k, &rt, 1.0, &Mat::default(), 0.0, &mut proj, 0)?;

        Ok(proj)
    }

    /// Check if a 3D point is in front of the camera
    fn is_in_front_of_camera(
        &self,
        point: &na::Point3<f64>,
        r: &na::Matrix3<f64>,
        t: &na::Vector3<f64>,
    ) -> bool {
        // Transform point to camera coordinates
        let point_cam = r * point + t;
        // Check if Z is positive (in front of camera)
        point_cam.z > 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_point_creation() {
        let pos = na::Point3::new(1.0, 2.0, 3.0);
        let point = MapPoint::new(pos, 0);
        assert_eq!(point.position, pos);
        assert_eq!(point.observations, 1);
        assert_eq!(point.id, 0);
        assert!(point.descriptor.is_none());
    }

    #[test]
    fn test_map_point_with_descriptor() {
        let pos = na::Point3::new(1.0, 2.0, 3.0);
        let desc = vec![1, 2, 3, 4];
        let point = MapPoint::with_descriptor(pos, desc.clone(), 5);
        assert_eq!(point.position, pos);
        assert_eq!(point.descriptor, Some(desc));
        assert_eq!(point.id, 5);
    }

    #[test]
    fn test_add_observation() {
        let mut point = MapPoint::new(na::Point3::new(0.0, 0.0, 1.0), 0);
        assert_eq!(point.observations, 1);
        point.add_observation();
        assert_eq!(point.observations, 2);
    }

    #[test]
    fn test_triangulator_creation() {
        let cam = CameraIntrinsics::kitti();
        let triangulator = Triangulator::new(cam);
        assert_eq!(triangulator.min_parallax_deg, 1.0);
        assert_eq!(triangulator.max_reproj_error, 4.0);
    }

    #[test]
    fn test_triangulator_builder() {
        let cam = CameraIntrinsics::kitti();
        let triangulator = Triangulator::new(cam)
            .with_min_parallax(2.0)
            .with_max_reproj_error(5.0);
        assert_eq!(triangulator.min_parallax_deg, 2.0);
        assert_eq!(triangulator.max_reproj_error, 5.0);
    }

    #[test]
    fn test_triangulate_empty_points() {
        let cam = CameraIntrinsics::kitti();
        let triangulator = Triangulator::new(cam);

        let pose1 = (na::Matrix3::identity(), na::Vector3::zeros());
        let pose2 = (na::Matrix3::identity(), na::Vector3::new(1.0, 0.0, 0.0));

        let points1 = Vector::new();
        let points2 = Vector::new();

        let result = triangulator.triangulate(&pose1, &pose2, &points1, &points2);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);
    }

    #[test]
    fn test_triangulate_mismatched_points() {
        let cam = CameraIntrinsics::kitti();
        let triangulator = Triangulator::new(cam);

        let pose1 = (na::Matrix3::identity(), na::Vector3::zeros());
        let pose2 = (na::Matrix3::identity(), na::Vector3::new(1.0, 0.0, 0.0));

        let mut points1 = Vector::new();
        points1.push(Point2f::new(100.0, 100.0));

        let points2 = Vector::new();

        let result = triangulator.triangulate(&pose1, &pose2, &points1, &points2);
        assert!(result.is_err());
    }

    #[test]
    fn test_is_in_front_of_camera() {
        let cam = CameraIntrinsics::kitti();
        let triangulator = Triangulator::new(cam);

        let r = na::Matrix3::identity();
        let t = na::Vector3::zeros();

        let point_front = na::Point3::new(0.0, 0.0, 5.0);
        let point_behind = na::Point3::new(0.0, 0.0, -5.0);

        assert!(triangulator.is_in_front_of_camera(&point_front, &r, &t));
        assert!(!triangulator.is_in_front_of_camera(&point_behind, &r, &t));
    }

    #[test]
    fn test_triangulate_synthetic_points() {
        // Create a synthetic scenario: two cameras and a few 3D points
        let cam = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0);
        let triangulator = Triangulator::new(cam);

        // Camera 1 at origin looking along +Z
        let pose1 = (na::Matrix3::identity(), na::Vector3::zeros());

        // Camera 2 moved 1 meter to the right along X-axis
        let pose2 = (na::Matrix3::identity(), na::Vector3::new(1.0, 0.0, 0.0));

        // Create synthetic 3D points in front of both cameras
        let point_3d_1 = na::Point3::new(0.0, 0.0, 10.0); // Center, 10m away
        let point_3d_2 = na::Point3::new(2.0, 1.0, 10.0); // Offset point

        // Project 3D points to 2D for both cameras
        let project_point =
            |point: &na::Point3<f64>, r: &na::Matrix3<f64>, t: &na::Vector3<f64>| {
                let p_cam = r * point + t;
                let x = cam.fx * (p_cam.x / p_cam.z) + cam.cx;
                let y = cam.fy * (p_cam.y / p_cam.z) + cam.cy;
                Point2f::new(x as f32, y as f32)
            };

        let mut points1 = Vector::new();
        let mut points2 = Vector::new();

        points1.push(project_point(&point_3d_1, &pose1.0, &pose1.1));
        points1.push(project_point(&point_3d_2, &pose1.0, &pose1.1));

        points2.push(project_point(&point_3d_1, &pose2.0, &pose2.1));
        points2.push(project_point(&point_3d_2, &pose2.0, &pose2.1));

        // Triangulate
        let result = triangulator.triangulate(&pose1, &pose2, &points1, &points2);
        if let Err(e) = &result {
            eprintln!("Triangulation error: {}", e);
        }
        assert!(result.is_ok());

        let map_points = result.unwrap();
        assert!(
            map_points.len() > 0,
            "Should triangulate at least some points"
        );

        // Check that triangulated points are reasonable
        for point in &map_points {
            // Points should be in front of camera
            assert!(point.position.z > 0.0, "Point should be in front of camera");
            // Points should be roughly at the expected depth
            assert!(
                point.position.z > 5.0 && point.position.z < 15.0,
                "Point depth should be reasonable"
            );
        }
    }
}
