use opencv::{core::Mat, prelude::*};

/// Camera intrinsic parameters
#[derive(Debug, Clone, Copy)]
pub struct CameraIntrinsics {
    pub fx: f64, // focal length x
    pub fy: f64, // focal length y
    pub cx: f64, // principal point x
    pub cy: f64, // principal point y
}

impl CameraIntrinsics {
    /// Create a new camera intrinsics with the given parameters.
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64) -> Self {
        Self { fx, fy, cx, cy }
    }

    /// KITTI dataset default camera (grayscale camera 0)
    pub fn kitti() -> Self {
        Self {
            fx: 718.856,
            fy: 718.856,
            cx: 607.1928,
            cy: 185.2157,
        }
    }

    /// Generic webcam preset (640x480)
    pub fn webcam_vga() -> Self {
        Self {
            fx: 500.0,
            fy: 500.0,
            cx: 320.0,
            cy: 240.0,
        }
    }

    /// Convert to OpenCV 3x3 camera matrix
    pub fn to_matrix(&self) -> Result<Mat, Box<dyn std::error::Error>> {
        use opencv::core::CV_64F;
        let mut mat =
            Mat::new_rows_cols_with_default(3, 3, CV_64F, opencv::core::Scalar::all(0.0))?;

        *mat.at_2d_mut::<f64>(0, 0)? = self.fx;
        *mat.at_2d_mut::<f64>(1, 1)? = self.fy;
        *mat.at_2d_mut::<f64>(0, 2)? = self.cx;
        *mat.at_2d_mut::<f64>(1, 2)? = self.cy;
        *mat.at_2d_mut::<f64>(2, 2)? = 1.0;

        Ok(mat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_camera_creation() {
        let cam = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0);
        assert_eq!(cam.fx, 500.0);
        assert_eq!(cam.fy, 500.0);
        assert_eq!(cam.cx, 320.0);
        assert_eq!(cam.cy, 240.0);
    }

    #[test]
    fn test_kitti_preset() {
        let cam = CameraIntrinsics::kitti();
        assert!(cam.fx > 700.0);
        assert!(cam.fy > 700.0);
    }

    #[test]
    fn test_to_matrix() {
        let cam = CameraIntrinsics::new(100.0, 200.0, 50.0, 75.0);
        let mat = cam.to_matrix();
        assert!(mat.is_ok());
        let mat = mat.unwrap();
        assert_eq!(mat.rows(), 3);
        assert_eq!(mat.cols(), 3);
    }
}
