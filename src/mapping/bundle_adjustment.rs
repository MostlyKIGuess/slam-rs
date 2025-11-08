use crate::odometry::CameraIntrinsics;
use nalgebra as na;
// use nalgebra::{Cholesky, LU};
use nalgebra::LU;
use std::collections::HashMap;
use std::ops::SubAssign;

/// SE(3) Lie algebra operations for proper pose optimization
mod lie {
    use nalgebra as na;

    /// Convert angle-axis vector to rotation matrix (Rodrigues' formula)
    pub fn exp_map(omega: &na::Vector3<f64>) -> na::Matrix3<f64> {
        let theta = omega.norm();
        if theta < 1e-8 {
            return na::Matrix3::identity();
        }
        let w = omega / theta;
        let w_hat = na::Matrix3::new(0.0, -w[2], w[1], w[2], 0.0, -w[0], -w[1], w[0], 0.0);
        na::Matrix3::identity() + w_hat * theta.sin() + (w_hat * w_hat) * (1.0 - theta.cos())
    }
}

#[derive(Debug, Clone)]
pub struct Observation {
    pub keyframe_idx: usize,
    pub point_idx: usize,
    pub pixel: na::Point2<f64>,
}

impl Observation {
    pub fn new(keyframe_idx: usize, point_idx: usize, pixel: na::Point2<f64>) -> Self {
        Self {
            keyframe_idx,
            point_idx,
            pixel,
        }
    }
}

/// Huber cost function
fn huber_loss(residual: f64, delta: f64) -> f64 {
    let rsq = residual * residual;
    if rsq <= delta * delta {
        rsq
    } else {
        2.0 * delta * residual.abs() - delta * delta
    }
}

pub struct BundleAdjuster {
    intrinsics: CameraIntrinsics,
    max_iterations: usize,
    lambda: f64,
    min_error_change: f64,
    huber_delta: f64,
}

impl BundleAdjuster {
    pub fn new(intrinsics: CameraIntrinsics) -> Self {
        Self {
            intrinsics,
            max_iterations: 10,
            lambda: 1e-3,
            min_error_change: 1e-6,
            huber_delta: 2.0, // pixels
        }
    }

    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    pub fn with_lambda(mut self, lambda: f64) -> Self {
        self.lambda = lambda;
        self
    }

    pub fn with_huber_delta(mut self, delta: f64) -> Self {
        self.huber_delta = delta;
        self
    }

    fn project(
        &self,
        point: &na::Point3<f64>,
        r: &na::Matrix3<f64>,
        t: &na::Vector3<f64>,
    ) -> Option<na::Point2<f64>> {
        let p_cam = r * point + t;

        if p_cam.z <= 1e-6 {
            return None;
        }

        let x = self.intrinsics.fx * (p_cam.x / p_cam.z) + self.intrinsics.cx;
        let y = self.intrinsics.fy * (p_cam.y / p_cam.z) + self.intrinsics.cy;
        Some(na::Point2::new(x, y))
    }

    /// Compute Jacobians for pose (6 DOF) and point (3 DOF)
    fn compute_jacobians(
        &self,
        point: &na::Point3<f64>,
        r: &na::Matrix3<f64>,
        t: &na::Vector3<f64>,
    ) -> Option<(na::Matrix2x6<f64>, na::Matrix2x3<f64>)> {
        let p_cam = r * point + t;

        if p_cam.z <= 1e-6 {
            return None;
        }

        let z = p_cam.z;
        let z2 = z * z;
        let fx = self.intrinsics.fx;
        let fy = self.intrinsics.fy;

        // ∂pixel/∂p_cam (2×3)
        let j_proj = na::Matrix2x3::new(
            fx / z,
            0.0,
            -fx * p_cam.x / z2,
            0.0,
            fy / z,
            -fy * p_cam.y / z2,
        );

        // ∂pixel/∂point = ∂pixel/∂p_cam * ∂p_cam/∂point (2×3)
        let j_point = j_proj * r;

        // ∂p_cam/∂pose = [∂p_cam/∂ω, ∂p_cam/∂t] (3×6)
        let point_cam = r * point;
        let point_cam_cross = na::Matrix3::new(
            0.0,
            -point_cam[2],
            point_cam[1],
            point_cam[2],
            0.0,
            -point_cam[0],
            -point_cam[1],
            point_cam[0],
            0.0,
        );

        let mut j_pose = na::Matrix2x6::zeros();
        // Rotation part: del pixel/del w = del pixel/del p_cam * del p_cam/del w
        j_pose
            .fixed_view_mut::<2, 3>(0, 0)
            .copy_from(&(j_proj * (-point_cam_cross)));
        // Translation part: del pixel/del t = del pixel/del p_cam * del p_cam/del t
        j_pose
            .fixed_view_mut::<2, 3>(0, 3)
            .copy_from(&(j_proj * (-point_cam_cross)));

        Some((j_pose, j_point))
    }

    pub fn compute_total_error(
        &self,
        poses: &[(na::Matrix3<f64>, na::Vector3<f64>)],
        points: &[na::Point3<f64>],
        observations: &[Observation],
    ) -> f64 {
        let mut total_error = 0.0;
        let mut count = 0;

        for obs in observations {
            if obs.keyframe_idx >= poses.len() || obs.point_idx >= points.len() {
                continue;
            }

            let (r, t) = &poses[obs.keyframe_idx];
            let point = &points[obs.point_idx];

            if let Some(proj) = self.project(point, r, t) {
                let dx = proj.x - obs.pixel.x;
                let dy = proj.y - obs.pixel.y;
                let residual = (dx * dx + dy * dy).sqrt();
                total_error += huber_loss(residual, self.huber_delta);
                count += 1;
            }
        }

        if count > 0 { total_error } else { 0.0 }
    }

    /// Full sparse BA using Schur complement for pose-point marginalization
    pub fn optimize(
        &self,
        poses: &mut [(na::Matrix3<f64>, na::Vector3<f64>)],
        points: &mut [na::Point3<f64>],
        observations: &[Observation],
        fix_first_pose: bool,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        if observations.is_empty() {
            return Ok(0.0);
        }

        let mut prev_error = self.compute_total_error(poses, points, observations);
        let n_poses = poses.len();
        let n_points = points.len();

        for _iter in 0..self.max_iterations {
            // Build sparse blocks
            let mut h_pp: HashMap<usize, na::Matrix6<f64>> = HashMap::new();
            let mut h_ll: HashMap<usize, na::Matrix3<f64>> = HashMap::new();
            let mut h_pl: HashMap<(usize, usize), na::Matrix6x3<f64>> = HashMap::new();
            let mut b_p: HashMap<usize, na::Vector6<f64>> = HashMap::new();
            let mut b_l: HashMap<usize, na::Vector3<f64>> = HashMap::new();

            // Build blocks
            for obs in observations {
                if obs.keyframe_idx >= n_poses || obs.point_idx >= n_points {
                    continue;
                }
                let (r, t) = &poses[obs.keyframe_idx];
                let point = &points[obs.point_idx];
                if let Some(proj) = self.project(point, r, t) {
                    let residual = na::Vector2::new(proj.x - obs.pixel.x, proj.y - obs.pixel.y);
                    let r_norm = residual.norm();
                    // Huber weight calculation
                    let weight = if r_norm > 1e-8 {
                        let huber_w = huber_loss(r_norm, self.huber_delta) / (r_norm * r_norm);
                        huber_w.sqrt()
                    } else {
                        1.0
                    };
                    let weighted_residual = residual * weight;

                    if let Some((j_pose, j_point)) = self.compute_jacobians(point, r, t) {
                        let j_pose_w = j_pose * weight;
                        let j_point_w = j_point * weight;

                        *h_pp
                            .entry(obs.keyframe_idx)
                            .or_insert_with(na::Matrix6::<f64>::zeros) +=
                            j_pose_w.transpose() * j_pose;
                        *h_ll
                            .entry(obs.point_idx)
                            .or_insert_with(na::Matrix3::<f64>::zeros) +=
                            j_point_w.transpose() * j_point;
                        *h_pl
                            .entry((obs.keyframe_idx, obs.point_idx))
                            .or_insert_with(na::Matrix6x3::<f64>::zeros) +=
                            j_pose_w.transpose() * j_point;

                        *b_p.entry(obs.keyframe_idx)
                            .or_insert_with(na::Vector6::<f64>::zeros) -=
                            j_pose_w.transpose() * weighted_residual;
                        *b_l.entry(obs.point_idx)
                            .or_insert_with(na::Vector3::<f64>::zeros) -=
                            j_point_w.transpose() * weighted_residual;
                    }
                }
            }

            // Build reduced pose system: h_reduced = h_pp - summation of H_pl * h_ll^-1 * H_pl^T
            let mut h_reduced = na::DMatrix::<f64>::zeros(n_poses * 6, n_poses * 6);
            let mut b_reduced = na::DVector::<f64>::zeros(n_poses * 6);

            // Initialize with h_pp and b_p
            for (i, h_pp_i) in &h_pp {
                let start = *i * 6;
                h_reduced.view_mut((start, start), (6, 6)).copy_from(h_pp_i);
            }
            for (i, b_p_i) in &b_p {
                let start = *i * 6;
                b_reduced.rows_mut(start, 6).copy_from(b_p_i);
            }

            // Fix first pose if requested
            if fix_first_pose && n_poses > 0 {
                h_reduced.view_mut((0, 0), (6, 6)).fill(0.0);
                h_reduced.view_mut((0, 0), (6, 6)).fill_with_identity();
                b_reduced.rows_mut(0, 6).fill(0.0);
            }

            // For each point, subtract its contribution from reduced system
            for j in 0..n_points {
                if let Some(h_ll_j) = h_ll.get(&j) {
                    let h_ll_inv = h_ll_j
                        .try_inverse()
                        .unwrap_or_else(|| na::Matrix3::<f64>::identity() * 1e6);

                    // Subtract h_pl[i,j] * h_ll^-1 * h_pl[i,j]^T from each pose block
                    for ((i, pj), h_pl_ij) in h_pl.iter() {
                        if *pj == j {
                            let contrib = h_pl_ij * h_ll_inv * h_pl_ij.transpose();
                            let start = *i * 6;
                            h_reduced
                                .view_mut((start, start), (6, 6))
                                .sub_assign(&contrib);
                        }
                    }

                    // Subtract H_pl[i,j] * H_ll^-1 * b_l[j] from each pose's b vector
                    if let Some(b_l_j) = b_l.get(&j) {
                        for ((i, pj), h_pl_ij) in h_pl.iter() {
                            if *pj == j {
                                let update = h_pl_ij * h_ll_inv * b_l_j;
                                let start = *i * 6;
                                b_reduced.rows_mut(start, 6).sub_assign(&update);
                            }
                        }
                    }
                }
            }

            // Apply damping directly to diagonal blocks for stability
            for i in 0..n_poses {
                for j in 0..6 {
                    h_reduced[(i * 6 + j, i * 6 + j)] += self.lambda * 10.0; // this is very strong damping, this needs testing
                }
            }

            // Solve reduced system with Cholesky
            // Didn't work lmao??
            // let chol = H_reduced
            //     .clone()
            //     .cholesky()
            //     .ok_or("Cholesky decomposition failed")?;
            // let delta_poses = chol.solve(&b_reduced);
            // LDLT doesn't seem to exists in Rust :(
            // let chol = Cholesky::new(h_reduced.clone())
            //     .ok_or("Cholesky failed - matrix not positive definite. Try increasing lambda")?;
            // let delta_poses = chol.solve(&b_reduced);
            // Solve reduced system with LU
            let lu = LU::new(h_reduced.clone());
            let delta_poses = lu.solve(&b_reduced).ok_or("LU solve failed")?;

            // Update poses
            for i in 0..n_poses {
                let delta = delta_poses.rows(i * 6, 6);
                let delta_rot = lie::exp_map(&na::Vector3::new(delta[0], delta[1], delta[2]));
                let delta_trans = na::Vector3::new(delta[3], delta[4], delta[5]);

                poses[i].0 = delta_rot * poses[i].0;
                poses[i].1 += delta_trans;
            }

            // Back-substitute to get point updates: del x_l = h_ll^-1 * (b_l - h_pl^T * del x_p)
            for j in 0..n_points {
                if let Some(h_ll_j) = h_ll.get(&j) {
                    let h_ll_inv = h_ll_j
                        .try_inverse()
                        .unwrap_or_else(|| na::Matrix3::<f64>::identity() * 1e6);

                    let mut sum = na::Vector3::<f64>::zeros();
                    for ((i, pj), h_pl_ij) in h_pl.iter() {
                        if *pj == j {
                            let delta_p = delta_poses.rows(*i * 6, 6);
                            sum += h_pl_ij.transpose() * delta_p;
                        }
                    }

                    if let Some(b_l_j) = b_l.get(&j) {
                        let delta_l = h_ll_inv * (b_l_j - sum);
                        points[j].coords += delta_l;
                    }
                }
            }

            // Check for divergence and abort if error increases
            let current_error = self.compute_total_error(poses, points, observations);
            if current_error > prev_error * 1.5 {
                // Optimization diverged, return previous error
                return Ok(prev_error);
            }
            let error_change = (prev_error - current_error).abs();

            if error_change < self.min_error_change {
                break;
            }

            prev_error = current_error;
        }

        Ok(prev_error)
    }

    pub fn local_bundle_adjustment(
        &self,
        poses: &mut [(na::Matrix3<f64>, na::Vector3<f64>)],
        points: &mut [na::Point3<f64>],
        observations: &[Observation],
        window_size: usize,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        if poses.is_empty() {
            return Ok(0.0);
        }

        let start_idx = poses.len().saturating_sub(window_size);
        let local_observations: Vec<Observation> = observations
            .iter()
            .filter(|obs| obs.keyframe_idx >= start_idx)
            .cloned()
            .collect();

        self.optimize(poses, points, &local_observations, start_idx == 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_se3_jacobian_numerical() {
        let cam = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0);
        let ba = BundleAdjuster::new(cam);

        let point = na::Point3::new(1.0, 2.0, 10.0);
        let r = na::Matrix3::identity();
        let t = na::Vector3::new(0.1, -0.2, 0.05);

        let (j_pose, _) = ba.compute_jacobians(&point, &r, &t).unwrap();

        // Numerical check: perturb rotation
        let eps = 1e-6;
        let delta_omega = na::Vector3::new(eps, 0.0, 0.0);
        let r_perturbed = lie::exp_map(&delta_omega) * r;

        let proj_original = ba.project(&point, &r, &t).unwrap();
        let proj_perturbed = ba.project(&point, &r_perturbed, &t).unwrap();

        let expected_change = j_pose.fixed_view::<2, 3>(0, 0) * delta_omega;
        let actual_change = proj_perturbed - proj_original;

        assert_relative_eq!(expected_change, actual_change, epsilon = 1e-5);
    }

    #[test]
    fn test_rotation_convergence() {
        let cam = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0);
        let ba = BundleAdjuster::new(cam).with_max_iterations(30);

        let true_point = na::Point3::new(1.0, 0.5, 5.0);
        let true_r = na::UnitQuaternion::from_euler_angles(0.1, 0.2, 0.05).to_rotation_matrix();
        let true_t = na::Vector3::new(0.1, -0.1, 0.0);

        let true_r_mat: na::Matrix3<f64> = true_r.into();
        let proj = ba.project(&true_point, &true_r_mat, &true_t).unwrap();

        let mut poses = vec![(na::Matrix3::identity(), na::Vector3::zeros())];
        let mut points = vec![na::Point3::new(1.5, 0.8, 6.0)];
        let observations = vec![Observation::new(0, 0, proj)];

        let result = ba.optimize(&mut poses, &mut points, &observations, false);
        assert!(result.is_ok());

        let final_error = result.unwrap();
        assert!(final_error < 1e-6, "Should reach near-zero error");
    }
}
