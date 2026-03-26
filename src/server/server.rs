use serde::{Deserialize, Serialize};
use std::sync::Arc;
use warp::{Filter, Reply, Rejection};
use std::convert::Infallible;

use crate::inference::inference::InferenceModel;

#[derive(Debug, Deserialize)]
pub struct PredictionRequest {
    pub distance: f32,
    pub price: f32,
    pub user_rating: f32,
}

#[derive(Debug, Deserialize)]
pub struct BatchPredictionRequest {
    pub inputs: Vec<(f32, f32, f32)>,
}

#[derive(Debug, Serialize)]
pub struct PredictionResponse {
    pub probability: f32,
    pub accepted: bool,
}

#[derive(Debug, Serialize)]
pub struct BatchPredictionResponse {
    pub predictions: Vec<PredictionResponse>,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

pub struct PredictionServer {
    inference_model: Arc<InferenceModel>,
}

impl PredictionServer {
    pub fn new(inference_model: InferenceModel) -> Self {
        Self {
            inference_model: Arc::new(inference_model),
        }
    }

    pub fn routes(&self) -> impl Filter<Extract = impl Reply, Error = Rejection> + Clone {
        let cors = warp::cors()
            .allow_any_origin()
            .allow_headers(vec!["content-type"]);

        self.health_check()
            .or(self.predict_single())
            .or(self.predict_batch())
            .or(self.get_info())
            .with(cors)
            .with(warp::log("api"))
    }

    fn health_check(&self) -> impl Filter<Extract = impl Reply, Error = Rejection> + Clone {
        warp::path("health")
            .and(warp::get())
            .map(|| {
                warp::reply::json(&serde_json::json!({
                    "status": "healthy",
                    "service": "acceptance-predictor"
                }))
            })
    }

    fn predict_single(&self) -> impl Filter<Extract = impl Reply, Error = Rejection> + Clone {
        let model = self.inference_model.clone();
        
        warp::path("predict")
            .and(warp::post())
            .and(warp::body::json())
            .and(warp::any().map(move || model.clone()))
            .and_then(|request: PredictionRequest, model: Arc<InferenceModel>| async move {
                let result = handle_single_prediction(request, model).await;
                Ok::<_, Infallible>(result)
            })
            .map(|reply| reply)
            .map_err(|_: Infallible| warp::reject::not_found())
    }

    fn predict_batch(&self) -> impl Filter<Extract = impl Reply, Error = Rejection> + Clone {
        let model = self.inference_model.clone();
        
        warp::path("predict_batch")
            .and(warp::post())
            .and(warp::body::json())
            .and(warp::any().map(move || model.clone()))
            .and_then(|request: BatchPredictionRequest, model: Arc<InferenceModel>| async move {
                let result = handle_batch_prediction(request, model).await;
                Ok::<_, Infallible>(result)
            })
            .map(|reply| reply)
            .map_err(|_: Infallible| warp::reject::not_found())
    }

    fn get_info(&self) -> impl Filter<Extract = impl Reply, Error = Rejection> + Clone {
        warp::path("info")
            .and(warp::get())
            .map(|| {
                warp::reply::json(&serde_json::json!({
                    "service": "acceptance-predictor",
                    "version": "0.1.0",
                    "description": "Ride acceptance prediction API",
                    "endpoints": {
                        "health": "GET /health - Health check",
                        "predict": "POST /predict - Single prediction",
                        "predict_batch": "POST /predict_batch - Batch prediction",
                        "info": "GET /info - Service information"
                    },
                    "input_format": {
                        "single": {
                            "distance": "f32",
                            "price": "f32", 
                            "user_rating": "f32"
                        },
                        "batch": {
                            "inputs": "[(f32, f32, f32)]"
                        }
                    }
                }))
            })
    }
}

async fn handle_single_prediction(
    request: PredictionRequest,
    model: Arc<InferenceModel>,
) -> Result<impl Reply, Infallible> {
    let probability = model.predict(request.distance, request.price, request.user_rating);
    let accepted = model.predict_acceptance(request.distance, request.price, request.user_rating);
    
    let response = PredictionResponse {
        probability,
        accepted,
    };
    
    Ok(warp::reply::json(&response))
}

async fn handle_batch_prediction(
    request: BatchPredictionRequest,
    model: Arc<InferenceModel>,
) -> Result<impl Reply, Infallible> {
    let probabilities = model.predict_batch(&request.inputs);
    let acceptances = model.predict_acceptance_batch(&request.inputs);
    
    let predictions: Vec<PredictionResponse> = probabilities
        .into_iter()
        .zip(acceptances.into_iter())
        .map(|(prob, accepted)| PredictionResponse {
            probability: prob,
            accepted,
        })
        .collect();
    
    let response = BatchPredictionResponse { predictions };
    Ok(warp::reply::json(&response))
}

pub async fn run_server(server: PredictionServer, port: u16) {
    let routes = server.routes();
    
    println!("🚀 Starting server on http://0.0.0.0:{}", port);
    println!("📊 Available endpoints:");
    println!("  GET  /health - Health check");
    println!("  POST /predict - Single prediction");
    println!("  POST /predict_batch - Batch prediction");
    println!("  GET  /info - Service information");
    
    warp::serve(routes)
        .run(([0, 0, 0, 0], port))
        .await;
}
