import bcrypt
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from sqlalchemy.orm import Session
from repositories.repositories import UserRepository, ScanRepository
from utils.jwt_util import generate_token

class UserService:
    @staticmethod
    def register_user(db: Session, email: str, name: str, password: str) -> Dict:
        """
        Register a new user
        """
        # Check if user already exists
        existing_user = UserRepository.get_user_by_email(db, email)
        if existing_user:
            return {"success": False, "error": "User already exists", "status_code": 409}
        
        # Hash password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        # Create user
        try:
            user = UserRepository.create_user(db, email, name, hashed_password)
            
            # Generate token
            token = generate_token(email, user.id)
            
            return {
                "success": True,
                "data": {
                    "token": token,
                    "name": user.name,
                    "email": user.email
                },
                "status_code": 201
            }
        except Exception as e:
            return {"success": False, "error": str(e), "status_code": 500}
    
    @staticmethod
    def login_user(db: Session, email: str, password: str) -> Dict:
        """
        Login a user
        """
        # Get user by email
        user = UserRepository.get_user_by_email(db, email)
        
        # Check if user exists and password matches
        if not user or not bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
            return {"success": False, "error": "Invalid credentials", "status_code": 401}
        
        # Generate token
        token = generate_token(email, user.id)
        
        return {
            "success": True,
            "data": {
                "token": token,
                "name": user.name,
                "email": user.email
            },
            "status_code": 200
        }
    
    @staticmethod
    def get_user_by_id(db: Session, user_id: int) -> Optional[Dict]:
        """
        Get a user by ID
        """
        user = UserRepository.get_user_by_id(db, user_id)
        if not user:
            return None
        
        return user.to_dict()

class ScanService:
    @staticmethod
    def create_scan(db: Session, job_id: str, file_name: str, user_email: Optional[str] = None) -> Dict:
        """
        Create a new scan record
        """
        user_id = None
        if user_email:
            user = UserRepository.get_user_by_email(db, user_email)
            if user:
                user_id = user.id
        
        try:
            scan = ScanRepository.create_scan(db, job_id, file_name, user_id)
            return {
                "success": True, 
                "data": scan.to_dict(),
                "status_code": 201
            }
        except Exception as e:
            return {"success": False, "error": str(e), "status_code": 500}
    
    @staticmethod
    def update_scan(db: Session, job_id: str, status: Optional[str] = None, 
                   metastasis_count: Optional[int] = None, 
                   total_volume: Optional[float] = None, 
                   metastasis_volumes: Optional[List[float]] = None) -> Dict:
        """
        Update a scan record
        """
        scan = ScanRepository.get_scan_by_job_id(db, job_id)
        if not scan:
            return {"success": False, "error": "Scan not found", "status_code": 404}
        
        try:
            if metastasis_count is not None:
                scan = ScanRepository.update_scan_results(
                    db, job_id, metastasis_count, total_volume, metastasis_volumes
                )
            elif status is not None:
                scan = ScanRepository.update_scan_status(db, job_id, status)
                
            return {
                "success": True,
                "data": scan.to_dict(),
                "status_code": 200
            }
        except Exception as e:
            return {"success": False, "error": str(e), "status_code": 500}
    
    @staticmethod
    def get_user_scans(db: Session, user_id: int, page: int = 1, per_page: int = 10) -> Dict:
        """
        Get scans for a user with pagination
        """
        offset = (page - 1) * per_page
        
        try:
            scans = ScanRepository.get_user_scans(db, user_id, offset, per_page)
            total_scans = ScanRepository.count_user_scans(db, user_id)
            
            return {
                "success": True,
                "data": {
                    "scans": [scan.to_dict() for scan in scans],
                    "pagination": {
                        "total_items": total_scans,
                        "total_pages": (total_scans + per_page - 1) // per_page,
                        "current_page": page,
                        "per_page": per_page
                    }
                },
                "status_code": 200
            }
        except Exception as e:
            return {"success": False, "error": str(e), "status_code": 500}
    
    @staticmethod
    def filter_scans(db: Session, user_id: int, filters: Dict[str, Any], page: int = 1, per_page: int = 10) -> Dict:
        """
        Filter scans by various criteria
        """
        # Process date filters
        if 'start_date' in filters and filters['start_date']:
            try:
                filters['start_date'] = datetime.strptime(filters['start_date'], "%Y-%m-%d")
            except ValueError:
                return {"success": False, "error": "Invalid start_date format. Use YYYY-MM-DD", "status_code": 400}
        
        if 'end_date' in filters and filters['end_date']:
            try:
                filters['end_date'] = datetime.strptime(filters['end_date'] + " 23:59:59", "%Y-%m-%d %H:%M:%S")
            except ValueError:
                return {"success": False, "error": "Invalid end_date format. Use YYYY-MM-DD", "status_code": 400}
        
        # Process numeric filters
        numeric_filters = ['min_metastasis', 'max_metastasis', 'min_volume', 'max_volume']
        for filter_name in numeric_filters:
            if filter_name in filters and filters[filter_name] is not None:
                try:
                    filters[filter_name] = float(filters[filter_name])
                except ValueError:
                    return {"success": False, "error": f"Invalid {filter_name} value", "status_code": 400}
        
        offset = (page - 1) * per_page
        
        try:
            scans = ScanRepository.filter_user_scans(db, user_id, filters, offset, per_page)
            total_scans = ScanRepository.count_filtered_scans(db, user_id, filters)
            
            return {
                "success": True,
                "data": {
                    "scans": [scan.to_dict() for scan in scans],
                    "pagination": {
                        "total_items": total_scans,
                        "total_pages": (total_scans + per_page - 1) // per_page,
                        "current_page": page,
                        "per_page": per_page
                    }
                },
                "status_code": 200
            }
        except Exception as e:
            return {"success": False, "error": str(e), "status_code": 500}
