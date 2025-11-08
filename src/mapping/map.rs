use nalgebra as na;
use std::collections::HashMap;

use super::triangulation::MapPoint;
use crate::odometry::CameraIntrinsics;

/// A global map containing all 3D points and their observations
pub struct Map {
    /// All map points indexed by ID
    points: HashMap<usize, MapPoint>,
    /// Next available point ID
    next_id: usize,
    /// Camera intrinsics for reprojection
    intrinsics: CameraIntrinsics,
    /// Minimum observations before a point is considered stable
    min_observations: usize,
}

impl Map {
    /// Create a new empty map
    pub fn new(intrinsics: CameraIntrinsics) -> Self {
        Self {
            points: HashMap::new(),
            next_id: 0,
            intrinsics,
            min_observations: 2,
        }
    }

    /// Add new points to the map
    pub fn add_points(&mut self, mut points: Vec<MapPoint>) {
        for point in points.iter_mut() {
            point.id = self.next_id;
            self.points.insert(self.next_id, point.clone());
            self.next_id += 1;
        }
    }

    /// Get all points in the map
    pub fn points(&self) -> Vec<&MapPoint> {
        self.points.values().collect()
    }

    /// Get number of points
    pub fn size(&self) -> usize {
        self.points.len()
    }

    /// Project a 3D point to 2D image coordinates
    fn project_point(
        &self,
        point: &na::Point3<f64>,
        r: &na::Matrix3<f64>,
        t: &na::Vector3<f64>,
    ) -> Option<na::Point2<f64>> {
        // Transform to camera frame
        let p_cam = r * point + t;

        // Check if point is in front of camera
        if p_cam.z <= 0.0 {
            return None;
        }

        // Project to image
        let x = self.intrinsics.fx * (p_cam.x / p_cam.z) + self.intrinsics.cx;
        let y = self.intrinsics.fy * (p_cam.y / p_cam.z) + self.intrinsics.cy;

        Some(na::Point2::new(x, y))
    }

    /// Find map points visible in current frame and match with features
    pub fn find_matches(
        &mut self,
        _keypoints: &opencv::core::Vector<opencv::core::KeyPoint>,
        descriptors: &opencv::core::Mat,
        pose: &(na::Matrix3<f64>, na::Vector3<f64>),
        matcher: &mut crate::FeatureMatcher,
    ) -> Result<Vec<(usize, usize)>, Box<dyn std::error::Error>> {
        let (r, t) = pose;
        let mut matches = Vec::new();

        // Build descriptor matrix for visible map points
        let mut visible_points = Vec::new();
        let mut visible_descriptors = Vec::new();

        for (id, point) in &self.points {
            // Project point to image
            if let Some(proj) = self.project_point(&point.position, r, t) {
                // Check if projection is within image bounds (rough check)
                if proj.x >= 0.0 && proj.x < 4000.0 && proj.y >= 0.0 && proj.y < 3000.0 {
                    if let Some(ref desc) = point.descriptor {
                        visible_points.push(*id);
                        visible_descriptors.push(desc.clone());
                    }
                }
            }
        }

        if visible_descriptors.is_empty() {
            return Ok(matches);
        }

        // Create Mat from visible descriptors
        let map_desc = descriptors_to_mat(&visible_descriptors)?;

        // Match current features against visible map points
        let raw_matches = matcher.match_descriptors(&map_desc, descriptors)?;
        let good_matches = matcher.filter_good_matches(&raw_matches, 2.0);

        // Convert to (map_point_id, keypoint_idx)
        for m in good_matches.iter() {
            let map_id = visible_points[m.query_idx as usize];
            let kp_idx = m.train_idx as usize;
            matches.push((map_id, kp_idx));
        }

        Ok(matches)
    }

    /// Update observations for matched points
    pub fn update_observations(&mut self, matches: &[(usize, usize)]) {
        for (map_id, _kp_idx) in matches {
            if let Some(point) = self.points.get_mut(map_id) {
                point.add_observation();
            }
        }
    }

    /// Prune bad points (few observations or high reprojection error)
    pub fn prune_outliers(&mut self) -> usize {
        let before = self.points.len();

        self.points
            .retain(|_id, point| point.observations >= self.min_observations);

        before - self.points.len()
    }

    /// Get points with minimum observations (stable points)
    pub fn stable_points(&self) -> Vec<&MapPoint> {
        self.points
            .values()
            .filter(|p| p.observations >= self.min_observations)
            .collect()
    }

    /// Clear all points
    pub fn clear(&mut self) {
        self.points.clear();
        self.next_id = 0;
    }
}

/// Helper to convert Vec<Vec<u8>> to OpenCV Mat
fn descriptors_to_mat(
    descriptors: &[Vec<u8>],
) -> Result<opencv::core::Mat, Box<dyn std::error::Error>> {
    use opencv::core::{CV_8U, Mat};
    use opencv::prelude::*;

    if descriptors.is_empty() {
        return Ok(Mat::default());
    }

    let rows = descriptors.len() as i32;
    let cols = descriptors[0].len() as i32;
    let mut mat =
        Mat::new_rows_cols_with_default(rows, cols, CV_8U, opencv::core::Scalar::all(0.0))?;

    for (i, desc) in descriptors.iter().enumerate() {
        for (j, &val) in desc.iter().enumerate() {
            *mat.at_2d_mut::<u8>(i as i32, j as i32)? = val;
        }
    }

    Ok(mat)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_creation() {
        let cam = CameraIntrinsics::kitti();
        let map = Map::new(cam);
        assert_eq!(map.size(), 0);
    }

    #[test]
    fn test_add_points() {
        let cam = CameraIntrinsics::kitti();
        let mut map = Map::new(cam);

        let points = vec![
            MapPoint::new(na::Point3::new(1.0, 2.0, 10.0), 0),
            MapPoint::new(na::Point3::new(2.0, 3.0, 10.0), 1),
        ];

        map.add_points(points);
        assert_eq!(map.size(), 2);
    }

    #[test]
    fn test_projection() {
        let cam = CameraIntrinsics::kitti();
        let map = Map::new(cam);

        let point = na::Point3::new(0.0, 0.0, 10.0);
        let r = na::Matrix3::identity();
        let t = na::Vector3::zeros();

        let proj = map.project_point(&point, &r, &t);
        assert!(proj.is_some());

        let p = proj.unwrap();
        assert!((p.x - cam.cx).abs() < 1e-6);
        assert!((p.y - cam.cy).abs() < 1e-6);
    }

    #[test]
    fn test_prune_outliers() {
        let cam = CameraIntrinsics::kitti();
        let mut map = Map::new(cam);

        let mut points = vec![
            MapPoint::new(na::Point3::new(1.0, 2.0, 10.0), 0),
            MapPoint::new(na::Point3::new(2.0, 3.0, 10.0), 1),
        ];
        points[0].observations = 5;
        points[1].observations = 1;

        map.add_points(points);
        let removed = map.prune_outliers();

        assert_eq!(removed, 1);
        assert_eq!(map.size(), 1);
    }
}
