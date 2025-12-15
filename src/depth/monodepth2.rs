use opencv::core::Scalar;
use opencv::{core::Mat, prelude::*};
use std::path::Path;

#[cfg(feature = "depth")]
use tch::{CModule, Device, IValue, Tensor};

/// MonoDepth2 depth estimator struct
#[cfg(feature = "depth")]
pub struct MonoDepth2 {
    encoder: CModule,
    decoder: CModule,
    device: Device,
    width: i32,
    height: i32,
}

#[cfg(feature = "depth")]
impl MonoDepth2 {
    /// Create a new MonoDepth2 estimator
    ///
    /// Arguments
    /// - `encoder_path` - Path to encoder.pt model file
    /// - `decoder_path` - Path to depth.pt (decoder) model file
    /// - `use_cuda` - Whether to use CUDA GPU acceleration
    /// - `width` - Input image width (default: 640)
    /// - `height` - Input image height (default: 192)
    pub fn new<P: AsRef<Path>>(
        encoder_path: P,
        decoder_path: P,
        use_cuda: bool,
        width: i32,
        height: i32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let device = if use_cuda && tch::Cuda::is_available() {
            Device::Cuda(0)
        } else {
            Device::Cpu
        };

        let encoder = CModule::load(encoder_path)?;
        let decoder = CModule::load(decoder_path)?;

        Ok(Self {
            encoder,
            decoder,
            device,
            width,
            height,
        })
    }

    /// Predict depth map from RGB image
    ///
    /// Arguments
    /// - `image` - Input BGR image (OpenCV Mat)
    ///
    /// Returns
    /// Depth map as OpenCV Mat (CV_32F, single channel, normalized 0-1)
    pub fn predict(&self, image: &Mat) -> Result<Mat, Box<dyn std::error::Error>> {
        // OpenCV Mat to Tensor
        let tensor = self.mat_to_tensor(image)?;

        // encoder
        let encoder_input = IValue::Tensor(tensor.unsqueeze(0));
        let encoder_output = self.encoder.forward_is(&[encoder_input])?;

        // Extract encoder features and run decoder
        // The decoder expects a SINGLE argument that is a list of 5 feature tensors
        let decoder_output = match &encoder_output {
            IValue::Tuple(tensors) => {
                if tensors.len() != 5 {
                    return Err(
                        format!("Expected 5 encoder features, got {}", tensors.len()).into(),
                    );
                }
                // Convert tuple to TensorList and pass as single argument
                let feature_tensors: Vec<Tensor> = tensors
                    .iter()
                    .filter_map(|iv| match iv {
                        IValue::Tensor(t) => Some(t.shallow_clone()),
                        _ => None,
                    })
                    .collect();
                if feature_tensors.len() != 5 {
                    return Err("Failed to extract 5 tensors from encoder tuple".into());
                }
                let features_list = IValue::TensorList(feature_tensors);
                self.decoder.forward_is(&[features_list])?
            }
            IValue::GenericList(list) => {
                if list.len() != 5 {
                    return Err(format!("Expected 5 encoder features, got {}", list.len()).into());
                }
                // Convert GenericList to TensorList and pass as single argument
                let feature_tensors: Vec<Tensor> = list
                    .iter()
                    .filter_map(|iv| match iv {
                        IValue::Tensor(t) => Some(t.shallow_clone()),
                        _ => None,
                    })
                    .collect();
                if feature_tensors.len() != 5 {
                    return Err("Failed to extract 5 tensors from encoder list".into());
                }
                let features_list = IValue::TensorList(feature_tensors);
                self.decoder.forward_is(&[features_list])?
            }
            IValue::TensorList(list) => {
                if list.len() != 5 {
                    return Err(format!("Expected 5 encoder features, got {}", list.len()).into());
                }
                // Decoder expects TensorList as single argument
                // Clone the tensors to create a new TensorList
                let feature_tensors: Vec<Tensor> = list.iter().map(|t| t.shallow_clone()).collect();
                let features_list = IValue::TensorList(feature_tensors);
                self.decoder.forward_is(&[features_list])?
            }
            IValue::Object(_) => {
                return Err("Encoder returned Object - unsupported format".into());
            }
            IValue::Tensor(_) => {
                return Err("Encoder returned single tensor instead of feature list".into());
            }
            IValue::GenericDict(_) => {
                return Err("Encoder returned dict instead of feature list".into());
            }
            IValue::Int(_) => {
                return Err("Encoder returned int instead of feature list".into());
            }
            IValue::Double(_) => {
                return Err("Encoder returned double instead of feature list".into());
            }
            IValue::Bool(_) => {
                return Err("Encoder returned bool instead of feature list".into());
            }
            IValue::String(_) => {
                return Err("Encoder returned string instead of feature list".into());
            }
            _ => {
                return Err("Unexpected encoder output format (unknown IValue type)".into());
            }
        };

        // Extract depth tensor - handle Tuple, GenericList, GenericDict, or TensorList
        // The decoder wrapper returns a list of 4 disparity maps at different scales
        // Index 0 is the highest resolution (full res)
        let depth_tensor = match decoder_output {
            IValue::Tuple(ref tensors) => {
                if tensors.is_empty() {
                    return Err("Decoder returned empty tuple".into());
                }
                match &tensors[0] {
                    IValue::Tensor(t) => t,
                    _ => return Err("Decoder output tuple[0] is not a tensor".into()),
                }
            }
            IValue::TensorList(ref list) => {
                if list.is_empty() {
                    return Err("Decoder returned empty TensorList".into());
                }
                // First tensor is the disparity map at full resolution
                &list[0]
            }
            IValue::GenericList(ref list) => {
                if list.is_empty() {
                    return Err("Decoder returned empty GenericList".into());
                }
                // Extract first tensor from GenericList
                match &list[0] {
                    IValue::Tensor(t) => t,
                    _ => return Err("Decoder output GenericList[0] is not a tensor".into()),
                }
            }
            IValue::GenericDict(ref dict) => {
                // MonoDepth2 decoder returns dict with key ("disp", 0)
                for (key, value) in dict.iter() {
                    if let IValue::Tuple(key_tuple) = key {
                        if key_tuple.len() == 2 {
                            if let IValue::String(s) = &key_tuple[0] {
                                if s.as_str() == "disp" {
                                    if let IValue::Tensor(t) = value {
                                        return Ok(self.process_depth_tensor(t)?);
                                    }
                                }
                            }
                        }
                    }
                }
                return Err("Could not find depth tensor in decoder dict output".into());
            }
            ref other => {
                return Err(format!(
                    "Unexpected decoder output format: {:?} (expected Tuple, TensorList, GenericList, or GenericDict)",
                    std::mem::discriminant(other)
                ).into());
            }
        };

        self.process_depth_tensor(depth_tensor)
    }

    /// Process depth tensor: normalize and convert to Mat
    /// Note: MonoDepth2 outputs disparity (inverse depth), not depth.
    /// Higher disparity = closer objects, lower disparity = farther objects.
    fn process_depth_tensor(
        &self,
        depth_tensor: &Tensor,
    ) -> Result<Mat, Box<dyn std::error::Error>> {
        // Debug: print tensor info
        let shape = depth_tensor.size();
        let min_val: f64 = depth_tensor.min().double_value(&[]);
        let max_val: f64 = depth_tensor.max().double_value(&[]);
        let mean_val: f64 = depth_tensor.mean(tch::Kind::Float).double_value(&[]);

        eprintln!("[DEBUG] Disparity tensor shape: {:?}", shape);
        eprintln!(
            "[DEBUG] Disparity range: min={:.6}, max={:.6}, mean={:.6}",
            min_val, max_val, mean_val
        );

        // Check if values are in expected range [0, 1] (after sigmoid in decoder)
        if min_val < 0.0 || max_val > 1.0 {
            eprintln!("[WARNING] Disparity values outside expected [0, 1] range!");
        }

        // MonoDepth2 outputs disparity (inverse depth)
        // Normalize disparity to 0-1 range for visualization
        let disp_min = depth_tensor.min();
        let disp_max = depth_tensor.max();
        let disp_range = &disp_max - &disp_min;

        // Avoid division by zero - extract scalar value from tensor
        let range_val: f64 = disp_range.double_value(&[]);
        eprintln!(
            "[DEBUG] Disparity range for normalization: {:.6}",
            range_val
        );

        let disp_normalized = if range_val > 1e-8 {
            (depth_tensor - &disp_min) / disp_range
        } else {
            eprintln!("[WARNING] Disparity range too small, returning zeros");
            depth_tensor.zeros_like()
        };

        // back to OpenCV Mat
        let depth_mat = self.tensor_to_mat(&disp_normalized)?;

        Ok(depth_mat)
    }

    /// Predict depth and return as colored visualization using magma.png
    ///
    /// Arguments
    /// - `image` - Input BGR image
    /// - `magma_path` - Path to magma colormap PNG file
    ///
    /// Returns
    /// Colored disparity map (CV_8UC3) using magma colormap.
    /// Closer objects appear brighter/warmer, farther objects appear darker/cooler.
    pub fn predict_colored(
        &self,
        image: &Mat,
        magma_path: &str,
    ) -> Result<Mat, Box<dyn std::error::Error>> {
        use opencv::{core, imgcodecs};

        let disparity = self.predict(image)?;

        // Load magma colormap PNG (it's 728x1 RGB)
        let magma = imgcodecs::imread(magma_path, imgcodecs::IMREAD_COLOR)?;
        if magma.empty() {
            return Err(format!("Failed to load magma colormap PNG from: {}", magma_path).into());
        }

        let (cm_width, cm_height) = (magma.cols(), magma.rows());
        let num_colors = cm_width.max(cm_height);

        // Collect all disparity values for percentile calculation
        // This matches the original MonoDepth2 visualization which uses 95th percentile
        let rows = disparity.rows();
        let cols = disparity.cols();
        let mut values: Vec<f32> = Vec::with_capacity((rows * cols) as usize);
        for y in 0..rows {
            for x in 0..cols {
                values.push(*disparity.at_2d::<f32>(y, x)?);
            }
        }

        // Sort to find percentiles
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let vmin = values.first().copied().unwrap_or(0.0);
        // Use 95th percentile as max to avoid outliers washing out the visualization
        let p95_idx = ((values.len() as f64) * 0.95) as usize;
        let vmax = values
            .get(p95_idx.min(values.len() - 1))
            .copied()
            .unwrap_or(1.0);

        eprintln!(
            "[DEBUG] Visualization: vmin={:.6}, vmax(95th percentile)={:.6}",
            vmin, vmax
        );

        let range = vmax - vmin;
        let range = if range < 1e-8 { 1.0 } else { range };

        // Create output colored Mat
        let mut colored = Mat::zeros(rows, cols, core::CV_8UC3)?.to_mat()?;

        // For each pixel, normalize using percentile range and map to colormap
        for y in 0..rows {
            for x in 0..cols {
                let val = *disparity.at_2d::<f32>(y, x)?;
                // Normalize using percentile range and clamp to [0, 1]
                let normalized = ((val - vmin) / range).clamp(0.0, 1.0);
                // Map to colormap index
                let idx = (normalized * (num_colors - 1) as f32) as i32;
                let idx_clamped = idx.clamp(0, num_colors - 1);

                let color = if cm_width > cm_height {
                    magma.at_2d::<core::Vec3b>(0, idx_clamped)?
                } else {
                    magma.at_2d::<core::Vec3b>(idx_clamped, 0)?
                };
                *colored.at_2d_mut::<core::Vec3b>(y, x)? = *color;
            }
        }

        Ok(colored)
    }

    /// Convert OpenCV Mat (BGR) to PyTorch Tensor (RGB, normalized)
    fn mat_to_tensor(&self, mat: &Mat) -> Result<Tensor, Box<dyn std::error::Error>> {
        use opencv::imgproc;

        // Resize if needed
        let mut resized = Mat::default();
        if mat.cols() != self.width || mat.rows() != self.height {
            opencv::imgproc::resize(
                mat,
                &mut resized,
                opencv::core::Size::new(self.width, self.height),
                0.0,
                0.0,
                imgproc::INTER_LINEAR,
            )?;
        } else {
            resized = mat.clone();
        }

        // Convert BGR to RGB
        let mut rgb = Mat::default();
        imgproc::cvt_color(
            &resized,
            &mut rgb,
            imgproc::COLOR_BGR2RGB,
            0,
            opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        // Convert to float tensor (H, W, C)
        let data_u8: Vec<u8> = rgb.data_bytes()?.to_vec();
        // convert to f32 because tch::Tensor::f_from_slice expects floats
        let data_f32: Vec<f32> = data_u8.iter().map(|&v| v as f32).collect();

        // create float tensor directly from slice. returns Result<Tensor, TchError>
        let tensor = tch::Tensor::f_from_slice(&data_f32)?
            .view([self.height as i64, self.width as i64, 3])
            .to_device(self.device);

        // Normalize to [0, 1] and permute to (C, H, W)
        let normalized = tensor / 255.0;
        let permuted = normalized.permute(&[2, 0, 1]);

        Ok(permuted)
    }

    /// Convert PyTorch Tensor (1, 1, H, W) to OpenCV Mat (H, W) CV_32F
    fn tensor_to_mat(&self, tensor: &Tensor) -> Result<Mat, Box<dyn std::error::Error>> {
        // Remove batch and channel dimensions
        let squeezed = tensor.squeeze_dim(0).squeeze_dim(0);

        // Move to CPU if needed
        let cpu_tensor = squeezed.to_device(Device::Cpu);

        // Get dimensions
        let shape = cpu_tensor.size();
        let height = shape[0] as i32;
        let _width = shape[1] as i32;

        // Convert to Vec<f32>
        let flat = cpu_tensor.f_view(-1)?;
        let numel = flat.numel();
        let mut data = vec![0f32; numel as usize];
        flat.f_copy_data(&mut data, numel)?;

        // Create OpenCV Mat from ref slice then reshape to (height, width)
        let mat_ref = Mat::from_slice(&data[..])?;
        let mat = mat_ref.reshape(1, height)?.try_clone()?;

        // ensure type is CV_32F. from_slice creates CV_32F for f32.
        Ok(mat)
    }

    /// Get expected input dimensions
    pub fn input_size(&self) -> (i32, i32) {
        (self.width, self.height)
    }

    /// Check for CUDA
    pub fn is_cuda(&self) -> bool {
        matches!(self.device, Device::Cuda(_))
    }
}

#[cfg(not(feature = "depth"))]
pub struct MonoDepth2;

#[cfg(not(feature = "depth"))]
impl MonoDepth2 {
    pub fn new<P>(
        _encoder_path: P,
        _decoder_path: P,
        _use_cuda: bool,
        _width: i32,
        _height: i32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Err("Depth estimation not enabled. Compile with --features depth".into())
    }
}

#[cfg(test)]
#[cfg(feature = "depth")]
mod tests {
    use super::*;

    #[test]
    fn test_monodepth2_with_manga_png() {
        // This test requires model files and manga.png
        let encoder_path = "weights/encoder.pt";
        let decoder_path = "weights/depth.pt";
        let test_image_path = "src/depth/manga.png";

        if !std::path::Path::new(encoder_path).exists()
            || !std::path::Path::new(decoder_path).exists()
            || !std::path::Path::new(test_image_path).exists()
        {
            println!("Skipping test: required files not found");
            return;
        }

        let model = MonoDepth2::new(encoder_path, decoder_path, false, 640, 192)
            .expect("Failed to create MonoDepth2 model");

        let image = opencv::imgcodecs::imread(test_image_path, opencv::imgcodecs::IMREAD_COLOR)
            .expect("Failed to load manga.png");
        assert!(!image.empty(), "Loaded image is empty");

        let depth = model.predict(&image).expect("Depth prediction failed");
        assert!(!depth.empty(), "Depth output is empty");
        assert_eq!(depth.rows(), 192, "Depth output height incorrect");
        assert_eq!(depth.cols(), 640, "Depth output width incorrect");
    }
}
