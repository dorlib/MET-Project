from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import logging
import time
from sqlalchemy.exc import OperationalError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration from environment variables
DB_USER = os.environ.get('DB_USER', 'root')
DB_PASSWORD = os.environ.get('DB_PASSWORD', 'password')
DB_HOST = os.environ.get('DB_HOST', 'mysql')
DB_PORT = os.environ.get('DB_PORT', '3306')
DB_NAME = os.environ.get('DB_NAME', 'met_user_service')

# Create connection string
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

logger.info(f"Database connecting to: {DB_HOST}:{DB_PORT}/{DB_NAME}")

# Create database engine with connection pool settings
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Test connections before use
    pool_recycle=3600,   # Recycle connections after an hour
    connect_args={"connect_timeout": 30}  # 30-second timeout for connections
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Function to get a database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Attempt to connect to the database with retries
def check_db_connection(max_retries=5, delay=2):
    """Check if database connection can be established with retries"""
    for i in range(max_retries):
        try:
            conn = engine.connect()
            conn.close()
            logger.info("Database connection successful")
            return True
        except OperationalError as e:
            if i < max_retries - 1:
                wait_time = delay * (i + 1)
                logger.warning(f"Database connection failed, retrying in {wait_time}s: {str(e)}")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to connect to database after {max_retries} attempts: {str(e)}")
                raise

# Initialize the database with tables
def init_db():
    # First check the database connection
    check_db_connection()
    
    # Import models and create tables
    from models.models import Base
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created")
