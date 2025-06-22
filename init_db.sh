#!/bin/bash
# Initialize the user service database for development

echo "Creating MySQL database and tables for user service..."

# Create database if it doesn't exist
mysql -h 127.0.0.1 -P 13306 -u root -ppassword << EOF
CREATE DATABASE IF NOT EXISTS met_user_service;
USE met_user_service;

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(100) NOT NULL UNIQUE,
    name VARCHAR(100) NOT NULL,
    password VARCHAR(200) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    role VARCHAR(20) DEFAULT 'user'
);

-- Create scans table
CREATE TABLE IF NOT EXISTS scans (
    id INT AUTO_INCREMENT PRIMARY KEY,
    job_id VARCHAR(36) NOT NULL UNIQUE,
    user_id INT,
    file_name VARCHAR(255) NOT NULL,
    status VARCHAR(20) DEFAULT 'processing',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    metastasis_count INT,
    total_volume FLOAT,
    metastasis_volumes VARCHAR(1000),
    FOREIGN KEY (user_id) REFERENCES users(id)
);
EOF

if [ $? -eq 0 ]; then
    echo "Database setup completed successfully."
else
    echo "Error setting up database."
    exit 1
fi
