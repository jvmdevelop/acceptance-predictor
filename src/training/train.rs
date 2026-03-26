use burn::{
    backend::{NdArray, ndarray::NdArrayDevice},
    config::Config,
    data::dataset::Dataset,
    tensor::Tensor,
    prelude::Backend,
};

use crate::{
    data::data_loader::RideLoader,
    model::model::RideModelConfig,
};

#[derive(Config)]
pub struct TrainingConfig {
    pub model: RideModelConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 32)]
    pub batch_size: usize,
    #[config(default = 1)]
    pub num_workers: usize,
    #[config(default = 0.1)]
    pub learning_rate: f64,
}

pub fn train(config: TrainingConfig, device: NdArrayDevice) {
    let model = config.model.init::<NdArray>(&device);

    let train_loader = RideLoader::from_csv("data/data.csv");
    
    println!("Starting training for {} epochs", config.num_epochs);

    for epoch in 1..=config.num_epochs {
        let model_train = model.clone();
        let mut loss_sum = 0.0;
        let mut batch_count = 0;

        for batch_idx in 0..(train_loader.len() / config.batch_size) {
            let batch = create_batch(&train_loader, batch_idx, config.batch_size, &device);
            
            let output = model_train.forward(batch.features.clone());
            let loss = binary_cross_entropy(output.clone(), batch.targets.clone());
            
            let loss_scalar = loss.clone().into_scalar();
            loss_sum += f32::from(loss_scalar);
            batch_count += 1;

            if batch_idx % 10 == 0 {
                println!("Epoch: {}, Batch: {}, Loss: {:.4}", epoch, batch_idx, loss_scalar);
            }
        }

        let avg_loss = loss_sum / batch_count as f32;
        println!("Epoch: {} completed, Average Loss: {:.4}", epoch, avg_loss);
    }

    println!("Training completed");
}

fn create_batch(
    loader: &RideLoader,
    batch_idx: usize,
    batch_size: usize,
    _device: &NdArrayDevice,
) -> RideBatch<NdArray> {
    let start_idx = batch_idx * batch_size;
    let mut features = Vec::new();
    let mut targets = Vec::new();

    for i in start_idx..(start_idx + batch_size).min(loader.len()) {
        if let Some(item) = loader.get(i) {
            features.push(item.features);
            targets.push(item.target);
        }
    }

    RideBatch {
        features: Tensor::stack(features, 0),
        targets: Tensor::stack(targets, 0),
    }
}

#[derive(Debug, Clone)]
pub struct RideBatch<B: Backend> {
    pub features: Tensor<B, 2>,
    pub targets: Tensor<B, 2>,
}

impl RideBatch<burn::backend::NdArray> {
    pub fn new(features: Tensor<burn::backend::NdArray, 2>, targets: Tensor<burn::backend::NdArray, 2>) -> Self {
        Self { features, targets }
    }
}

fn binary_cross_entropy(output: Tensor<NdArray, 2>, targets: Tensor<NdArray, 2>) -> Tensor<NdArray, 1> {
    let epsilon = 1e-7f32;
    let output_dims = output.dims();
    let output_device = output.device();
    
    let output_clamped = output.clamp(epsilon, 1.0 - epsilon);
    let ones = Tensor::ones(output_dims, &output_device);
    
    let term1 = targets.clone() * output_clamped.clone().log();
    let term2 = (ones.clone() - targets) * (ones - output_clamped).log();
    
    let loss = -(term1 + term2);
    loss.mean()
}
