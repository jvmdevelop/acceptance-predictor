use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    prelude::Backend,
    tensor::{Device, Tensor},
};

#[derive(Config, Debug)]
struct RideModelConfig {
    #[config(default = 3)]
    input_size: usize,
    #[config(default = 10)]
    hidden_size: usize,
    #[config(default = 1)]
    output_size: usize,
}

impl RideModelConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> RideModel<B> {
        RideModel {
            linear_1: LinearConfig::new(self.input_size, self.hidden_size).init(device),
            relu_1: Relu::new(),
            linear_2: LinearConfig::new(self.hidden_size, self.output_size).init(device),
        }
    }
}

#[derive(Module, Debug)]
struct RideModel<B: Backend> {
    linear_1: Linear<B>,
    relu_1: Relu,
    linear_2: Linear<B>,
}

impl<B: Backend> RideModel<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear_1.forward(x);
        let x = self.relu_1.forward(x);
        let x = self.linear_2.forward(x);
        x
    }
}
