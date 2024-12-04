CREATE DATABASE IF NOT EXISTS nyc_taxi_data;

USE nyc_taxi_data;

-- Table for storing the taxi data
CREATE TABLE trips (
    id INT AUTO_INCREMENT PRIMARY KEY,
    vendor_id VARCHAR(10),
    pickup_datetime DATETIME,
    dropoff_datetime DATETIME,
    passenger_count INT,
    trip_distance FLOAT,
    pickup_longitude FLOAT,
    pickup_latitude FLOAT,
    dropoff_longitude FLOAT,
    dropoff_latitude FLOAT,
    payment_type VARCHAR(20),
    fare_amount FLOAT,
    extra FLOAT,
    mta_tax FLOAT,
    tip_amount FLOAT,
    tolls_amount FLOAT,
    total_amount FLOAT
);
