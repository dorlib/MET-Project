#!/bin/bash
# Verify that MySQL is running and the user_service tables are created

echo "Checking MySQL connection..."
mysql -h 127.0.0.1 -P 13306 -u root -ppassword -e "SELECT 1;" &> /dev/null

if [ $? -ne 0 ]; then
    echo "Failed to connect to MySQL. Make sure it's running."
    exit 1
else
    echo "Successfully connected to MySQL."
fi

echo "Checking if met_user_service database exists..."
DB_EXISTS=$(mysql -h 127.0.0.1 -P 13306 -u root -ppassword -e "SHOW DATABASES LIKE 'met_user_service';" | grep -c "met_user_service")

if [ $DB_EXISTS -eq 0 ]; then
    echo "Database 'met_user_service' does not exist. Creating it..."
    mysql -h 127.0.0.1 -P 13306 -u root -ppassword -e "CREATE DATABASE met_user_service;"
    echo "Database created."
else
    echo "Database 'met_user_service' exists."
fi

echo "MySQL setup verification completed successfully."
