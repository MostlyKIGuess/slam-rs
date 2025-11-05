use opencv::{
    core::{KeyPoint, Mat, Vector},
    features2d::ORB,
    prelude::*,
};

/// Simple ORB feature detector wrapper
pub struct OrbDetector {
    orb: opencv::core::Ptr<ORB>,
    max_features: i32,
}

impl OrbDetector {
    /// new ORB detector with specified max features
    pub fn new(max_features: i32) -> Result<Self, Box<dyn std::error::Error>> {
        let orb = ORB::create_def()?;

        Ok(Self { orb, max_features })
    }

    pub fn detect(&mut self, image: &Mat) -> Result<Vector<KeyPoint>, Box<dyn std::error::Error>> {
        let mut keypoints = Vector::new();
        self.orb.detect(image, &mut keypoints, &Mat::default())?;
        Ok(keypoints)
    }

    pub fn detect_and_compute(
        &mut self,
        image: &Mat,
    ) -> Result<(Vector<KeyPoint>, Mat), Box<dyn std::error::Error>> {
        let mut keypoints = Vector::new();
        let mut descriptors = Mat::default();
        self.orb.detect_and_compute(
            image,
            &Mat::default(),
            &mut keypoints,
            &mut descriptors,
            false,
        )?;
        Ok((keypoints, descriptors))
    }

    pub fn max_features(&self) -> i32 {
        self.max_features
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orb_detector_creation() {
        let detector = OrbDetector::new(500);
        assert!(detector.is_ok());
        assert_eq!(detector.unwrap().max_features(), 500);
    }

    #[test]
    fn test_detect_on_blank_image() {
        let mut detector = OrbDetector::new(100).unwrap();
        let blank = Mat::zeros(480, 640, opencv::core::CV_8UC1)
            .unwrap()
            .to_mat()
            .unwrap();
        let keypoints = detector.detect(&blank).unwrap();
        assert_eq!(keypoints.len(), 0);
    }

    #[test]
    fn test_detect_and_compute() {
        let mut detector = OrbDetector::new(500).unwrap();
        let blank = Mat::zeros(480, 640, opencv::core::CV_8UC1)
            .unwrap()
            .to_mat()
            .unwrap();
        let result = detector.detect_and_compute(&blank);
        assert!(result.is_ok());
        let (keypoints, descriptors) = result.unwrap();
        assert_eq!(keypoints.len(), descriptors.rows() as usize);
    }
}
