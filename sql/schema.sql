-- Customer Churn Prediction Database Schema
-- Database initialization for MySQL

CREATE DATABASE IF NOT EXISTS churn_db;
USE churn_db;

-- Table 1: Model Prediction Logs
CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    customer_id VARCHAR(50) DEFAULT NULL,
    input_json JSON NOT NULL,
    tenure INT NOT NULL,
    contract VARCHAR(50) NOT NULL,
    monthly_charges DECIMAL(8,2) NOT NULL,
    total_charges DECIMAL(10,2) NOT NULL,
    churn_probability DECIMAL(5,4) NOT NULL,
    prediction_label VARCHAR(30) NOT NULL,
    risk_level VARCHAR(20) NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    notes TEXT DEFAULT NULL
);

-- Table 2: Model Training Run Metadata
CREATE TABLE IF NOT EXISTS model_runs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_name VARCHAR(50) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    roc_auc DECIMAL(5,4) NOT NULL,
    pr_auc DECIMAL(5,4) NOT NULL,
    f1_churn DECIMAL(5,4) NOT NULL,
    recall_churn DECIMAL(5,4) NOT NULL,
    precision_churn DECIMAL(5,4) NOT NULL,
    accuracy DECIMAL(5,4) NOT NULL,
    threshold DECIMAL(4,3) NOT NULL,
    hyperparameters JSON DEFAULT NULL,
    notes TEXT DEFAULT NULL
);
