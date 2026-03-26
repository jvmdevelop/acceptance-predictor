use crate::model::model::{RideModel, RideModelConfig};
use burn::{
    backend::{NdArray, ndarray::NdArrayDevice},
    tensor::{Tensor, activation::sigmoid},
};

pub struct InferenceModel {
    model: RideModel<NdArray>,
    device: NdArrayDevice,
}

impl InferenceModel {
    pub fn new(config: RideModelConfig, device: NdArrayDevice) -> Self {
        let model = config.init::<NdArray>(&device);
        Self { model, device }
    }

    pub fn predict(&self, distance: f32, price: f32, user_rating: f32) -> f32 {
        let features = self.preprocess_input(distance, price, user_rating);
        let output = self.model.forward(features);
        output.into_scalar().into()
    }

    pub fn predict_batch(&self, inputs: &[(f32, f32, f32)]) -> Vec<f32> {
        let features = self.preprocess_batch(inputs);
        let output = self.model.forward(features);
        output.into_data().value
    }

    pub fn predict_acceptance(&self, distance: f32, price: f32, user_rating: f32) -> bool {
        self.predict(distance, price, user_rating) > 0.5
    }

    pub fn predict_acceptance_batch(&self, inputs: &[(f32, f32, f32)]) -> Vec<bool> {
        self.predict_batch(inputs)
            .into_iter()
            .map(|p| p > 0.5)
            .collect()
    }

    fn preprocess_input(&self, distance: f32, price: f32, user_rating: f32) -> Tensor<NdArray, 2> {
        Tensor::from_floats(
            [[distance / 50.0, price / 1000.0, (user_rating - 1.0) / 4.0]],
            &self.device,
        )
    }

    fn preprocess_batch(&self, inputs: &[(f32, f32, f32)]) -> Tensor<NdArray, 2> {
        let rows: Vec<Tensor<NdArray, 1>> = inputs
            .iter()
            .map(|(distance, price, user_rating)| {
                Tensor::from_floats(
                    [distance / 50.0, price / 1000.0, (user_rating - 1.0) / 4.0],
                    &self.device,
                )
            })
            .collect();

        Tensor::stack(rows, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_model() -> InferenceModel {
        InferenceModel::new(RideModelConfig::new(), NdArrayDevice::Cpu)
    }

    #[test]
    fn test_single_prediction() {
        let p = make_model().predict(10.0, 500.0, 4.5);
        assert!((0.0..=1.0).contains(&p));
    }

    #[test]
    fn test_batch_prediction() {
        let preds = make_model().predict_batch(&[(10.0, 500.0, 4.5), (50.0, 2000.0, 2.0)]);
        assert_eq!(preds.len(), 2);
        assert!(preds.iter().all(|&p| (0.0..=1.0).contains(&p)));
    }

    #[test]
    fn test_acceptance_prediction() {
        let _ = make_model().predict_acceptance(10.0, 500.0, 4.5);
    }
}
