"""
MET Segmentation Service Entry Point
"""
import logging
from app.main import app

if __name__ == "__main__":
    # Configure logging for direct execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
