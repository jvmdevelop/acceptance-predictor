use burn::{
    backend::{NdArray, ndarray::NdArrayDevice},
    data::dataloader::{DataLoaderBuilder, batcher::Batcher},
    prelude::Backend,
    tensor::Tensor,
};

use crate::data::data_loader::{RideItem, RideLoader};

pub struct BatchConfig {
    pub batch_size: usize,
    pub shuffle: Option<u64>,
    pub num_workers: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            shuffle: Some(42),
            num_workers: 4,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RideBatcher {
    device: NdArrayDevice,
}

impl RideBatcher {
    pub fn new(device: NdArrayDevice) -> Self {
        Self { device }
    }

    pub fn batch(&self, items: Vec<RideItem>) -> Option<RideBatch<NdArray>> {
        if items.is_empty() {
            return None;
        }

        let mut features = Vec::with_capacity(items.len());
        let mut targets = Vec::with_capacity(items.len());

        for item in items {
            features.push(item.features);
            targets.push(item.target);
        }

        Some(RideBatch {
            features: Tensor::stack(features, 0).to_device(&self.device),
            targets: Tensor::stack(targets, 0).to_device(&self.device),
        })
    }
}

impl Batcher<RideItem, RideBatch<NdArray>> for RideBatcher {
    fn batch(&self, items: Vec<RideItem>) -> RideBatch<NdArray> {
        self.batch(items).unwrap()
    }
}

pub fn create_dataloader(
    config: BatchConfig,
) -> DataLoaderBuilder<RideItem, RideBatch<NdArray<f32>>> {
    let batcher = RideBatcher::new(NdArrayDevice::Cpu);

    let mut builder = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers);

    if let Some(seed) = config.shuffle {
        builder = builder.shuffle(seed);
    }

    builder
}

#[derive(Debug, Clone)]
pub struct RideBatch<B: Backend> {
    pub features: Tensor<B, 2>,
    pub targets: Tensor<B, 2>,
}

impl<B: Backend> RideBatch<B> {
    pub fn new(features: Tensor<B, 2>, targets: Tensor<B, 2>) -> Self {
        Self { features, targets }
    }

    pub fn features(&self) -> Tensor<B, 2> {
        self.features.clone()
    }

    pub fn targets(&self) -> Tensor<B, 2> {
        self.targets.clone()
    }

    pub fn batch_size(&self) -> usize {
        self.features.dims()[0]
    }

    pub fn split(&self, ratio: f32) -> (Self, Self) {
        let batch_size = self.features.dims()[0];
        let split_size = (batch_size as f32 * ratio) as usize;

        let train_features = self.features.clone().slice([0..split_size]);
        let val_features = self.features.clone().slice([split_size..batch_size]);
        let train_targets = self.targets.clone().slice([0..split_size]);
        let val_targets = self.targets.clone().slice([split_size..batch_size]);

        (
            RideBatch {
                features: train_features,
                targets: train_targets,
            },
            RideBatch {
                features: val_features,
                targets: val_targets,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::data_generator::generate_data;
    use crate::data::data_loader::RideLoader;

    #[test]
    fn test_batch_creation() {
        let data = generate_data(10);
        let device = NdArrayDevice::Cpu;

        let mut items = Vec::new();
        for i in 0..data.len() {
            let features = Tensor::from_floats(
                [
                    data[i].distance / 50.0,
                    data[i].price / 1000.0,
                    (data[i].user_rating - 1.0) / 4.0,
                ],
                &device,
            );
            let target = Tensor::from_floats([data[i].accepted as f32], &device);

            items.push(RideItem { features, target });
        }

        let batcher = RideBatcher::new(device);
        let batch = batcher.batch(items).unwrap();

        assert_eq!(batch.batch_size(), 10);
        assert_eq!(batch.features.dims()[1], 3);
        assert_eq!(batch.targets.dims()[1], 1);
    }

    #[test]
    fn test_batch_split() {
        let data = generate_data(10);
        let device = NdArrayDevice::Cpu;

        let mut items = Vec::new();
        for i in 0..data.len() {
            let features = Tensor::from_floats(
                [
                    data[i].distance / 50.0,
                    data[i].price / 1000.0,
                    (data[i].user_rating - 1.0) / 4.0,
                ],
                &device,
            );
            let target = Tensor::from_floats([data[i].accepted as f32], &device);

            items.push(RideItem { features, target });
        }

        let batcher = RideBatcher::new(device);
        let batch = batcher.batch(items).unwrap();
        let (train_batch, val_batch) = batch.split(0.8);

        assert_eq!(train_batch.batch_size(), 8);
        assert_eq!(val_batch.batch_size(), 2);
    }
}
