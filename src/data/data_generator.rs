use std::fs::File;

use rand::prelude::*;
use serde::{Deserialize, Serialize};

trait ToArray {
    fn to_array(&self) -> Vec<String>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Data {
    pub distance: f32,
    pub price: f32,
    pub user_rating: f32,
    pub accepted: i8,
}

impl ToArray for Data {
    fn to_array(&self) -> Vec<String> {
        vec![
            self.distance.to_string(),
            self.price.to_string(),
            self.user_rating.to_string(),
            self.accepted.to_string(),
        ]
    }
}

pub fn write(data: Vec<Data>) {
    let file = File::create("data/data.csv").expect("Unable to create file");
    let mut wtr = csv::Writer::from_writer(file);
    
    wtr.write_record(&["distance", "price", "user_rating", "accepted"])
        .expect("Unable to write header");
    
    for d in data {
        wtr.write_record(d.to_array())
            .expect("Unable to write data");
    }
    wtr.flush().expect("Unable to flush writer");
}

pub fn generate_data(size: usize) -> Vec<Data> {
    let mut rng = rand::rng();

    let mut data = Vec::with_capacity(size);

    for _ in 0..size {
        let distance: f32 = rng.random_range(0.0..100.0);
        let price: f32 = rng.random_range(0.0..4000.0);
        let user_rating: f32 = rng.random_range(0.0..5.0);

        let accepted = if price > (distance * 50.0) { 1 } else { 0 };

        data.push(Data {
            distance,
            price,
            user_rating,
            accepted,
        });
    }

    data
}

pub fn write_and_generate_data(size: usize) {
    let data = generate_data(size);
    write(data);
}

pub fn ensure_data_exists(size: usize) {
    if !std::path::Path::new("data/data.csv").exists() {
        write_and_generate_data(size);
    }
}
