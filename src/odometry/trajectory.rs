use nalgebra as na;
use serde::{Deserialize, Serialize};

/// Single trajectory point with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryPoint {
    pub frame: usize,
    pub position: [f64; 3],
    pub timestamp: f64,
}

/// Trajectory tracker for visual odometry
pub struct Trajectory {
    points: Vec<TrajectoryPoint>,
    global_pose: na::Matrix4<f64>,
}

impl Trajectory {
    /// Create a new trajectory starting at origin
    pub fn new() -> Self {
        Self {
            points: vec![TrajectoryPoint {
                frame: 0,
                position: [0.0, 0.0, 0.0],
                timestamp: 0.0,
            }],
            global_pose: na::Matrix4::identity(),
        }
    }

    /// Update pose with relative rotation and translation
    pub fn update(
        &mut self,
        rotation: &na::Matrix3<f64>,
        translation: &na::Vector3<f64>,
        frame: usize,
        timestamp: f64,
    ) {
        // Build relative transformation matrix
        let mut relative_transform = na::Matrix4::identity();

        // rotation
        for i in 0..3 {
            for j in 0..3 {
                relative_transform[(i, j)] = rotation[(i, j)];
            }
        }

        // translation
        relative_transform[(0, 3)] = translation[0];
        relative_transform[(1, 3)] = translation[1];
        relative_transform[(2, 3)] = translation[2];

        // Compose: T_global = T_global * T_relative
        self.global_pose = self.global_pose * relative_transform;

        // Extract current position
        let position = [
            self.global_pose[(0, 3)],
            self.global_pose[(1, 3)],
            self.global_pose[(2, 3)],
        ];

        self.points.push(TrajectoryPoint {
            frame,
            position,
            timestamp,
        });
    }

    /// Get current global pose
    pub fn current_pose(&self) -> &na::Matrix4<f64> {
        &self.global_pose
    }

    /// Get current rotation and translation as a tuple
    pub fn current_pose_rt(&self) -> (na::Matrix3<f64>, na::Vector3<f64>) {
        let mut rotation = na::Matrix3::zeros();
        for i in 0..3 {
            for j in 0..3 {
                rotation[(i, j)] = self.global_pose[(i, j)];
            }
        }

        let translation = na::Vector3::new(
            self.global_pose[(0, 3)],
            self.global_pose[(1, 3)],
            self.global_pose[(2, 3)],
        );

        (rotation, translation)
    }

    /// Get all trajectory points
    pub fn points(&self) -> &[TrajectoryPoint] {
        &self.points
    }

    /// Calculate total distance traveled
    pub fn total_distance(&self) -> f64 {
        let mut distance = 0.0;
        for i in 1..self.points.len() {
            let p1 = &self.points[i - 1].position;
            let p2 = &self.points[i].position;

            let dx = p2[0] - p1[0];
            let dy = p2[1] - p1[1];
            let dz = p2[2] - p1[2];

            distance += (dx * dx + dy * dy + dz * dz).sqrt();
        }
        distance
    }

    /// Get number of trajectory points
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Check if trajectory is empty
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Export trajectory to JSON
    pub fn to_json(&self) -> Result<String, Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(&self.points)?;
        Ok(json)
    }

    /// Save trajectory to file
    pub fn save_to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = self.to_json()?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trajectory_creation() {
        let traj = Trajectory::new();
        assert_eq!(traj.len(), 1);
        assert_eq!(traj.points()[0].frame, 0);
    }

    #[test]
    fn test_trajectory_update() {
        let mut traj = Trajectory::new();
        let r = na::Matrix3::identity();
        let t = na::Vector3::new(1.0, 0.0, 0.0);

        traj.update(&r, &t, 1, 0.1);

        assert_eq!(traj.len(), 2);
        assert_eq!(traj.points()[1].frame, 1);
        assert!((traj.points()[1].position[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_total_distance() {
        let mut traj = Trajectory::new();
        let r = na::Matrix3::identity();

        // Move 3 units in x direction
        let t1 = na::Vector3::new(3.0, 0.0, 0.0);
        traj.update(&r, &t1, 1, 0.1);

        // Move 4 units in y direction
        let t2 = na::Vector3::new(0.0, 4.0, 0.0);
        traj.update(&r, &t2, 2, 0.2);

        let distance = traj.total_distance();
        assert!((distance - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_json_export() {
        let mut traj = Trajectory::new();
        let r = na::Matrix3::identity();
        let t = na::Vector3::new(1.0, 2.0, 3.0);
        traj.update(&r, &t, 1, 0.1);

        let json = traj.to_json();
        assert!(json.is_ok());
        assert!(json.unwrap().contains("position"));
    }
}
