use anyhow::{Context, Result};
use image::{DynamicImage, ImageReader};
use ndarray::Array;
use ort::session::{Session};
use std::path::Path;
use std::time::Instant;

/// Preprocesses an image for ONNX model inference.
///
/// # Arguments
/// * `image_path` - Path to the input image
/// * `input_size` - Target size (width, height) for the model input
///
/// # Returns
/// A preprocessed ndarray ready for inference with shape (1, 3, height, width)
fn preprocess(image_path: &str, input_size: (u32, u32)) -> Result<Array<f32, ndarray::Ix4>> {
    // Load and convert image to RGB
    let img = ImageReader::open(image_path)
        .context("Failed to open image")?
        .decode()
        .context("Failed to decode image")?;

    // Resize to target size
    let img = img.resize_exact(input_size.0, input_size.1, image::imageops::FilterType::Triangle);

    // Convert to RGB if not already
    let img = match img {
        DynamicImage::ImageRgb8(rgb) => rgb,
        other => other.to_rgb8(),
    };

    let (width, height) = img.dimensions();
    
    // Create a 4D array with shape (1, 3, height, width)
    // Initialize with zeros
    let mut array = Array::<f32, _>::zeros((1, 3, height as usize, width as usize));

    // Fill the array with normalized pixel values
    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x, y);
            // Normalize to [0, 1] and arrange in CHW format
            array[[0, 0, y as usize, x as usize]] = pixel[0] as f32 / 255.0; // R
            array[[0, 1, y as usize, x as usize]] = pixel[1] as f32 / 255.0; // G
            array[[0, 2, y as usize, x as usize]] = pixel[2] as f32 / 255.0; // B
        }
    }

    Ok(array)
}

/// Compute softmax values for each set of scores
fn softmax_vec(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_values: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp_values.iter().sum();
    exp_values.iter().map(|&x| x / sum).collect()
}

fn main() -> Result<()> {
    // Enable ONNX Runtime verbose logging to see execution provider details
    std::env::set_var("ORT_LOG_SEVERITY_LEVEL", "1"); // 1 = INFO level

    // From https://huggingface.co/Xenova/resnet-152/blob/main/onnx/model.onnx
    // ResNet-152 trained on ImageNet-1k with ONNX saved in HuggingFace transformers compatible format
    let model_path = "../../../models/onnx/resnet-152.onnx";
    let image_dir = "..";
    let sample_images = vec!["cat.jpg", "dog.jpg"];

    // Key ImageNet-1k classes:
    // - 285: Egyptian Cat
    // - 226: Briard
    // Ref: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a

    // --- 1. Load the ONNX model ---
    if !Path::new(model_path).exists() {
        anyhow::bail!(
            "Error: Model not found at {}\nPlease ensure you have a trained 'resnet-152.onnx' in the 'models/onnx/' directory.",
            model_path
        );
    }

    println!("Loading model from {}...", model_path);

    // --- Configure session with CUDA (GPU) execution provider ONLY ---
    // Disable fallback to CPU by only specifying CUDA provider
    // If CUDA is not available, the session creation will fail instead of falling back to CPU
    let mut session = Session::builder()
        .context("Failed to create session builder")?
        .with_execution_providers([
            ort::execution_providers::CUDAExecutionProvider::default()
                .with_device_id(0)
                .build()
                .error_on_failure(),  // This will cause an error if CUDA provider fails to initialize
        ])
        .context("Failed to configure CUDA execution provider - ensure CUDA/GPU is available")?
        .commit_from_file(model_path)
        .context("Failed to load ONNX model")?;
    
    println!("✓ Model loaded successfully!");
    println!("✓ CUDA execution provider configured (GPU acceleration enabled)");
    println!("✓ CPU fallback is DISABLED - will fail if GPU is not available\n");

    // --- 2. Process each sample image ---
    for image_file in sample_images {
        let image_path = format!("{}/{}", image_dir, image_file);
        
        if !Path::new(&image_path).exists() {
            println!("\nWarning: Sample image not found at {}. Skipping.", image_path);
            continue;
        }

        println!("\n--- Processing {} ---", image_file);

        // --- 3. Preprocess the image ---
        let input_tensor = preprocess(&image_path, (224, 224))
            .context(format!("Failed to preprocess {}", image_file))?;

        // --- 4. Run inference ---
        println!("Running inference on GPU 100 times...");
        
        // Convert ndarray to Vec and create an ort Value with shape
        let shape = input_tensor.shape().to_vec();
        let mut last_outputs = None;
        let mut duration : std::time::Duration = std::time::Duration::new(0, 0);
        
        for _i in 0..100 {
            let data: Vec<f32> = input_tensor.iter().copied().collect();
            let input_value = ort::value::Value::from_array((shape.as_slice(), data))
                .context("Failed to create input value")?;
            
            let start = Instant::now();
            let outputs = session
                .run(ort::inputs!["pixel_values" => input_value])
                .context("Failed to run inference")?;
            let elapsed = start.elapsed();
            duration += elapsed;
            
            // Extract and store the output data to avoid lifetime issues
            let output_data = outputs["logits"]
                .try_extract_tensor::<f32>()
                .context("Failed to extract output tensor")?;
            let (_shape, logits_slice) = output_data;
            last_outputs = Some(logits_slice.to_vec());
        }

        println!("Inference completed 100 times in {:.2?}", duration);
        println!("Average time per inference: {:.2?}", duration / 100);

        // --- 5. Post-process the result ---
        let logits = last_outputs.unwrap();
        
        // Apply softmax to get probabilities
        let probabilities = softmax_vec(&logits);
        
        // Find the predicted class and confidence
        let (predicted_class_index, &confidence) = probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        println!(
            "Prediction: '{}' with {:.2}% confidence.",
            predicted_class_index,
            confidence * 100.0
        );
    }

    Ok(())
}
