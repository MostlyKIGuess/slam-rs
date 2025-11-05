use opencv::{
    core::{Mat, Vector},
    features2d::BFMatcher,
    prelude::*,
};

/// Feature matcher using Brute Force
/// For each descriptor in the first set, this matcher finds the closest descriptor in the second set by trying each one. This descriptor matcher supports masking permissible matches of descriptor sets.
pub struct FeatureMatcher {
    matcher: opencv::core::Ptr<BFMatcher>,
}

impl FeatureMatcher {
    /// Create a new feature matcher using Brute Force
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let matcher = BFMatcher::create(
            opencv::core::NORM_HAMMING,
            false, // don't cross check
        )?;
        Ok(Self { matcher })
    }

    /// Match descriptors between two frames
    pub fn match_descriptors(
        &mut self,
        desc1: &Mat,
        desc2: &Mat,
    ) -> Result<Vector<opencv::core::DMatch>, Box<dyn std::error::Error>> {
        if desc1.empty() || desc2.empty() {
            return Ok(Vector::new());
        }

        let mut matches = Vector::new();
        self.matcher
            .train_match(desc1, desc2, &mut matches, &Mat::default())?;
        Ok(matches)
    }

    /// Filter matches by distance (keep best matches)
    pub fn filter_good_matches(
        &self,
        matches: &Vector<opencv::core::DMatch>,
        ratio: f32,
    ) -> Vector<opencv::core::DMatch> {
        if matches.is_empty() {
            return Vector::new();
        }

        // Find min distance
        let mut min_dist = f32::MAX;
        for m in matches.iter() {
            if m.distance < min_dist {
                min_dist = m.distance;
            }
        }
        // Keep matches with distance < max(2*min_dist, 30.0)
        // TODO: Implement a better threshold calculation
        let threshold = (ratio * min_dist).max(30.0);
        let mut good = Vector::new();
        for m in matches.iter() {
            if m.distance < threshold {
                good.push(m);
            }
        }
        good
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matcher_creation() {
        let matcher = FeatureMatcher::new();
        assert!(matcher.is_ok());
    }

    #[test]
    fn test_empty_match() {
        let mut matcher = FeatureMatcher::new().unwrap();
        let empty = Mat::default();
        let result = matcher.match_descriptors(&empty, &empty);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);
    }
}
