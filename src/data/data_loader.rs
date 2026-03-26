use burn::{
    backend::{NdArray, ndarray::NdArrayDevice},
    data::dataset::Dataset,
};

use super::data_generator::Data;

#[derive(Debug, Clone)]
pub struct RideLoader {
    data: Vec<Data>,
}

impl RideLoader {
    pub fn from_csv(path: &str) -> Self {
        let data = read_data(path);
        Self { data }
    }
}

#[derive(Debug, Clone)]
pub struct RideItem {
    pub features: Vec<f32>,
    pub target: f32,
}

impl Dataset<RideItem> for RideLoader {
    fn len(&self) -> usize {
        self.data.len()
    }
    fn get(&self, index: usize) -> Option<RideItem> {
        let data = self.data.get(index)?;
        Some(RideItem {
            features: vec![
                data.distance / 50.0,
                data.price / 1000.0,
                (data.user_rating - 1.0) / 4.0,
            ],
            target: data.accepted as f32,
        })
    }
}

fn read_data(path: &str) -> Vec<Data> {
    let mut reader = csv::Reader::from_path(path).unwrap();
    reader.headers().unwrap();
    reader
        .deserialize()
        .collect::<Result<Vec<Data>, _>>()
        .unwrap()
}
