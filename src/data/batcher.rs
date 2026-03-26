use burn::{data::dataloader::batcher::Batcher, prelude::Backend, tensor::Tensor};

use crate::data::data_loader::RideItem;

#[derive(Clone)]
pub struct RideBatcher<B: Backend> {
    device: B::Device,
}
#[derive(Debug, Clone)]
pub struct RideBatch<B: Backend> {
    pub features: Tensor<B, 2>,
    pub targets: Tensor<B, 2>,
}

impl<B: Backend> RideBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<RideItem, RideBatch<B>> for RideBatcher<B> {
    fn batch(&self, items: Vec<RideItem>) -> RideBatch<B> {
        let features = items
            .iter()
            .map(|item| {
                let arr: [f32; 3] = item.features.as_slice().try_into().unwrap();
                Tensor::<B, 1>::from_floats(arr, &self.device)
            })
            .collect();

        let targets = items
            .iter()
            .map(|item| Tensor::<B, 1>::from_floats([item.target], &self.device))
            .collect();

        RideBatch {
            features: Tensor::stack(features, 0),
            targets: Tensor::stack(targets, 0),
        }
    }
}
