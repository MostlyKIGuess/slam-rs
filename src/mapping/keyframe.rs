use nalgebra as na;

/// Keyframe selection criteria
#[derive(Debug, Clone)]
pub struct KeyframeConfig {
    /// Minimum translation distance to consider new keyframe (meters)
    pub min_translation: f64,
    /// Minimum rotation angle to consider new keyframe (radians)
    pub min_rotation: f64,
    /// Minimum ratio of matches compared to last keyframe
    pub min_match_ratio: f64,
    /// Maximum number of frames since last keyframe
    pub max_frames: usize,
}

impl Default for KeyframeConfig {
    fn default() -> Self {
        Self {
            min_translation: 0.1, // 10cm
            min_rotation: 0.1,    // ~5.7 degrees
            min_match_ratio: 0.8, // 80% of previous matches
            max_frames: 10,       // Force keyframe every 10 frames
        }
    }
}

/// Keyframe selector for visual odometry
pub struct KeyframeSelector {
    config: KeyframeConfig,
    frames_since_last: usize,
    last_keyframe_matches: usize,
}

impl KeyframeSelector {
    /// Create new keyframe selector with default config
    pub fn new() -> Self {
        Self::with_config(KeyframeConfig::default())
    }

    /// Create new keyframe selector with custom config
    pub fn with_config(config: KeyframeConfig) -> Self {
        Self {
            config,
            frames_since_last: 0,
            last_keyframe_matches: 0,
        }
    }

    /// Check if current frame should be a keyframe
    pub fn should_be_keyframe(
        &mut self,
        rotation: &na::Matrix3<f64>,
        translation: &na::Vector3<f64>,
        num_matches: usize,
    ) -> bool {
        self.frames_since_last += 1;

        // Force keyframe if too many frames passed
        if self.frames_since_last >= self.config.max_frames {
            self.mark_as_keyframe(num_matches);
            return true;
        }

        // Check translation
        let trans_norm = translation.norm();
        if trans_norm >= self.config.min_translation {
            self.mark_as_keyframe(num_matches);
            return true;
        }

        // Check rotation (convert to angle)
        let rotation_angle = rotation_matrix_to_angle(rotation);
        if rotation_angle >= self.config.min_rotation {
            self.mark_as_keyframe(num_matches);
            return true;
        }

        // Check match quality degradation
        if self.last_keyframe_matches > 0 {
            let match_ratio = num_matches as f64 / self.last_keyframe_matches as f64;
            if match_ratio < self.config.min_match_ratio {
                self.mark_as_keyframe(num_matches);
                return true;
            }
        }

        false
    }

    /// Reset selector state
    pub fn reset(&mut self) {
        self.frames_since_last = 0;
        self.last_keyframe_matches = 0;
    }

    /// Mark current frame as keyframe
    fn mark_as_keyframe(&mut self, num_matches: usize) {
        self.frames_since_last = 0;
        self.last_keyframe_matches = num_matches;
    }

    /// Get frames since last keyframe
    pub fn frames_since_last(&self) -> usize {
        self.frames_since_last
    }
}

/// Convert rotation matrix to rotation angle (radians)
fn rotation_matrix_to_angle(rotation: &na::Matrix3<f64>) -> f64 {
    // trace(R) = 1 + 2*cos(theta)
    let trace = rotation.trace();
    let cos_angle = (trace - 1.0) / 2.0;
    let cos_angle = cos_angle.clamp(-1.0, 1.0); // Numerical stability
    cos_angle.acos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keyframe_selector_creation() {
        let selector = KeyframeSelector::new();
        assert_eq!(selector.frames_since_last(), 0);
    }

    #[test]
    fn test_force_keyframe_after_max_frames() {
        let config = KeyframeConfig {
            max_frames: 5,
            ..Default::default()
        };
        let mut selector = KeyframeSelector::with_config(config);

        let r = na::Matrix3::identity();
        let t = na::Vector3::zeros();

        for i in 0..4 {
            assert!(!selector.should_be_keyframe(&r, &t, 100), "Frame {}", i);
        }
        assert!(
            selector.should_be_keyframe(&r, &t, 100),
            "Frame 5 should be keyframe"
        );
    }

    #[test]
    fn test_keyframe_on_large_translation() {
        let mut selector = KeyframeSelector::new();
        let r = na::Matrix3::identity();
        let t = na::Vector3::new(0.2, 0.0, 0.0); // 20cm movement

        assert!(selector.should_be_keyframe(&r, &t, 100));
    }

    #[test]
    fn test_keyframe_on_large_rotation() {
        let mut selector = KeyframeSelector::new();
        let angle: f64 = 0.15; // > 0.1 rad threshold
        let r = na::Matrix3::new(
            angle.cos(),
            -angle.sin(),
            0.0,
            angle.sin(),
            angle.cos(),
            0.0,
            0.0,
            0.0,
            1.0,
        );
        let t = na::Vector3::zeros();

        assert!(selector.should_be_keyframe(&r, &t, 100));
    }

    #[test]
    fn test_no_keyframe_small_motion() {
        let mut selector = KeyframeSelector::new();
        selector.mark_as_keyframe(100);

        let r = na::Matrix3::identity();
        let t = na::Vector3::new(0.01, 0.0, 0.0); // 1cm, below threshold

        assert!(!selector.should_be_keyframe(&r, &t, 95));
    }
}
