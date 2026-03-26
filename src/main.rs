mod batching;
mod data;
mod inference;
mod model;
mod training;

use burn::{
    backend::{NdArray, ndarray::NdArrayDevice},
};

use training::train::{TrainingConfig, train};
use data::data_generator::write_and_generate_data;
use inference::inference::InferenceModel;

fn main() {
    println!("Generating data...");
    write_and_generate_data(1000);
    
    println!("Initializing training...");
    let device = NdArrayDevice::Cpu;
    
    let config = TrainingConfig::new(model::model::RideModelConfig::new());
    
    train(config, device.clone());
    
    println!("\n=== Testing Inference ===");
    
    let inference_model = InferenceModel::new(model::model::RideModelConfig::new(), device);
    
    println!("Single predictions:");
    let test_cases = vec![
        (5.0, 200.0, 4.8),   
        (50.0, 3000.0, 2.1),  
        (15.0, 800.0, 3.5),   
    ];
    
    for (i, (distance, price, rating)) in test_cases.iter().enumerate() {
        let probability = inference_model.predict(*distance, *price, *rating);
        let acceptance = inference_model.predict_acceptance(*distance, *price, *rating);
        println!(
            "Case {}: Distance={:.1}, Price={:.1}, Rating={:.1} -> Probability: {:.3}, Accepted: {}",
            i + 1, distance, price, rating, probability, acceptance
        );
    }
    
    println!("\nBatch predictions:");
    let batch_inputs = vec![
        (10.0, 500.0, 4.5),
        (25.0, 1500.0, 3.0),
        (40.0, 2500.0, 1.5),
    ];
    
    let batch_probabilities = inference_model.predict_batch(&batch_inputs);
    let batch_acceptances = inference_model.predict_acceptance_batch(&batch_inputs);
    
    for (i, (((distance, price, rating), prob), accepted)) in batch_inputs.iter()
        .zip(batch_probabilities.iter())
        .zip(batch_acceptances.iter())
        .enumerate()
    {
        println!(
            "Batch {}: Distance={:.1}, Price={:.1}, Rating={:.1} -> Probability: {:.3}, Accepted: {}",
            i + 1, distance, price, rating, prob, accepted
        );
    }
    
    println!("\n=== Testing Inference Complete ===");
    println!("✅ Model training and inference testing completed successfully!");
    println!("📊 Ready for production use with the inference module.");
}
