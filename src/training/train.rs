use crate::{
    data::{batcher::RideBatcher, data_loader::RideLoader},
    model::model::RideModelConfig,
};
use burn::{
    backend::{Autodiff, NdArray, ndarray::NdArrayDevice},
    config::Config,
    data::dataloader::DataLoaderBuilder,
    nn::loss::BinaryCrossEntropyLossConfig,
    optim::{GradientsParams, Optimizer, SgdConfig},
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

type TrainBackend = Autodiff<NdArray>;

pub fn train(config: TrainingConfig, device: NdArrayDevice) {
    let mut model = config.model.init::<TrainBackend>(&device);
    let mut optim = SgdConfig::new().init();
    let loss_fn = BinaryCrossEntropyLossConfig::new().init(&device);

    let dataset = RideLoader::from_csv("data/data.csv");
    let batcher = RideBatcher::<TrainBackend>::new(device.clone());
    let loader = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .shuffle(42)
        .build(dataset);

    for epoch in 1..=config.num_epochs {
        let mut loss_sum = 0.0f32;
        let mut batch_count = 0.0f32;

        for (batch_idx, batch) in loader.iter().enumerate() {
            let output = model.forward(batch.features);
            let loss = loss_fn.forward(output, batch.targets.int());

            let loss_scalar: f32 = loss.clone().into_scalar().into();
            loss_sum += loss_scalar;
            batch_count += 1.0;

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(config.learning_rate, model, grads);

            if batch_idx % 10 == 0 {
                println!("Epoch {epoch}, Batch {batch_idx}, Loss: {loss_scalar:.4}");
            }
        }

        println!(
            "Epoch {epoch} done. Avg loss: {:.4}",
            loss_sum / batch_count
        );
    }
}
