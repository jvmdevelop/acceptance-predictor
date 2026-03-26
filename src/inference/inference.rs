use burn::{
    backend::{NdArray, ndarray::NdArrayDevice},
    tensor::Tensor,
};

use crate::model::model::{RideModel, RideModelConfig};

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
        let prediction = output.into_scalar();
        
        1.0 / (1.0 + (-prediction).exp())
    }

    pub fn predict_batch(&self, inputs: &[(f32, f32, f32)]) -> Vec<f32> {
        let features = self.preprocess_batch(inputs);
        let outputs = self.model.forward(features);
        
        let vec_output = outputs.to_data().value;
        vec_output.into_iter()
            .map(|x| 1.0 / (1.0 + (-x).exp()))
            .collect()
    }

    pub fn predict_acceptance(&self, distance: f32, price: f32, user_rating: f32) -> bool {
        let probability = self.predict(distance, price, user_rating);
        probability > 0.5
    }

    pub fn predict_acceptance_batch(&self, inputs: &[(f32, f32, f32)]) -> Vec<bool> {
        let probabilities = self.predict_batch(inputs);
        probabilities.into_iter().map(|p| p > 0.5).collect()
    }

    fn preprocess_input(&self, distance: f32, price: f32, user_rating: f32) -> Tensor<NdArray, 2> {
        let normalized_features = [
            distance / 50.0,
            price / 1000.0,
            (user_rating - 1.0) / 4.0,
        ];
        
        Tensor::from_floats([normalized_features], &self.device)
    }

    fn preprocess_batch(&self, inputs: &[(f32, f32, f32)]) -> Tensor<NdArray, 2> {
        let normalized_features: Vec<f32> = inputs
            .iter()
            .flat_map(|(distance, price, user_rating)| {
                [
                    distance / 50.0,
                    price / 1000.0,
                    (user_rating - 1.0) / 4.0,
                ]
            })
            .collect();
        
        Tensor::from_floats(normalized_features.as_slice(), &self.device)
            .reshape([inputs.len(), 3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_prediction() {
        let config = RideModelConfig::new();
        let device = NdArrayDevice::Cpu;
        let inference_model = InferenceModel::new(config, device);
        
        let prediction = inference_model.predict(10.0, 500.0, 4.5);
        assert!(prediction >= 0.0 && prediction <= 1.0);
    }

    #[test]
    fn test_batch_prediction() {
        let config = RideModelConfig::new();
        let device = NdArrayDevice::Cpu;
        let inference_model = InferenceModel::new(config, device);
        
        let inputs = vec![(10.0, 500.0, 4.5), (50.0, 2000.0, 2.0)];
        let predictions = inference_model.predict_batch(&inputs);
        
        assert_eq!(predictions.len(), 2);
        for pred in predictions {
            assert!(pred >= 0.0 && pred <= 1.0);
        }
    }

    #[test]
    fn test_acceptance_prediction() {
        let config = RideModelConfig::new();
        let device = NdArrayDevice::Cpu;
        let inference_model = InferenceModel::new(config, device);
        
        let acceptance = inference_model.predict_acceptance(10.0, 500.0, 4.5);
        assert!(acceptance == true || acceptance == false);
    }
}
