import json
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from models.models import User, Scan

class UserRepository:
    @staticmethod
    def create_user(db: Session, email: str, name: str, hashed_password: str) -> User:
        """
        Create a new user in the database
        """
        user = User(email=email, name=name, password=hashed_password)
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    
    @staticmethod
    def get_user_by_email(db: Session, email: str) -> Optional[User]:
        """
        Get a user by email
        """
        return db.query(User).filter(User.email == email).first()
    
    @staticmethod
    def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
        """
        Get a user by ID
        """
        return db.query(User).filter(User.id == user_id).first()
    
    @staticmethod
    def get_all_users(db: Session, skip: int = 0, limit: int = 100) -> List[User]:
        """
        Get all users with pagination
        """
        return db.query(User).offset(skip).limit(limit).all()

class ScanRepository:
    @staticmethod
    def create_scan(db: Session, job_id: str, file_name: str, user_id: Optional[int] = None) -> Scan:
        """
        Create a new scan record
        """
        scan = Scan(job_id=job_id, file_name=file_name, user_id=user_id)
        db.add(scan)
        db.commit()
        db.refresh(scan)
        return scan
    
    @staticmethod
    def get_scan_by_job_id(db: Session, job_id: str) -> Optional[Scan]:
        """
        Get a scan by job ID
        """
        return db.query(Scan).filter(Scan.job_id == job_id).first()
    
    @staticmethod
    def update_scan_status(db: Session, job_id: str, status: str) -> Optional[Scan]:
        """
        Update scan status
        """
        scan = db.query(Scan).filter(Scan.job_id == job_id).first()
        if scan:
            scan.status = status
            db.commit()
            db.refresh(scan)
        return scan
    
    @staticmethod
    def update_scan_results(db: Session, job_id: str, metastasis_count: int, 
                           total_volume: float, metastasis_volumes: List[float]) -> Optional[Scan]:
        """
        Update scan with analysis results
        """
        scan = db.query(Scan).filter(Scan.job_id == job_id).first()
        if scan:
            scan.status = 'completed'
            scan.metastasis_count = metastasis_count
            scan.total_volume = total_volume
            scan.metastasis_volumes = json.dumps(metastasis_volumes)
            db.commit()
            db.refresh(scan)
        return scan
    
    @staticmethod
    def get_user_scans(db: Session, user_id: int, skip: int = 0, limit: int = 10) -> List[Scan]:
        """
        Get all scans for a user with pagination
        """
        return db.query(Scan).filter(Scan.user_id == user_id).order_by(Scan.created_at.desc()).offset(skip).limit(limit).all()
    
    @staticmethod
    def count_user_scans(db: Session, user_id: int) -> int:
        """
        Count total scans for a user
        """
        return db.query(Scan).filter(Scan.user_id == user_id).count()
    
    @staticmethod
    def filter_user_scans(db: Session, user_id: int, filters: Dict[str, Any], skip: int = 0, limit: int = 10) -> List[Scan]:
        """
        Filter scans by various criteria
        """
        query = db.query(Scan).filter(Scan.user_id == user_id)
        
        # Apply filters
        if 'min_metastasis' in filters and filters['min_metastasis'] is not None:
            query = query.filter(Scan.metastasis_count >= filters['min_metastasis'])
            
        if 'max_metastasis' in filters and filters['max_metastasis'] is not None:
            query = query.filter(Scan.metastasis_count <= filters['max_metastasis'])
            
        if 'min_volume' in filters and filters['min_volume'] is not None:
            query = query.filter(Scan.total_volume >= filters['min_volume'])
            
        if 'max_volume' in filters and filters['max_volume'] is not None:
            query = query.filter(Scan.total_volume <= filters['max_volume'])
            
        if 'start_date' in filters and filters['start_date'] is not None:
            query = query.filter(Scan.created_at >= filters['start_date'])
            
        if 'end_date' in filters and filters['end_date'] is not None:
            query = query.filter(Scan.created_at <= filters['end_date'])
        
        # Order by newest first and paginate
        query = query.order_by(Scan.created_at.desc()).offset(skip).limit(limit)
        
        return query.all()
    
    @staticmethod
    def count_filtered_scans(db: Session, user_id: int, filters: Dict[str, Any]) -> int:
        """
        Count filtered scans
        """
        query = db.query(Scan).filter(Scan.user_id == user_id)
        
        # Apply filters
        if 'min_metastasis' in filters and filters['min_metastasis'] is not None:
            query = query.filter(Scan.metastasis_count >= filters['min_metastasis'])
            
        if 'max_metastasis' in filters and filters['max_metastasis'] is not None:
            query = query.filter(Scan.metastasis_count <= filters['max_metastasis'])
            
        if 'min_volume' in filters and filters['min_volume'] is not None:
            query = query.filter(Scan.total_volume >= filters['min_volume'])
            
        if 'max_volume' in filters and filters['max_volume'] is not None:
            query = query.filter(Scan.total_volume <= filters['max_volume'])
            
        if 'start_date' in filters and filters['start_date'] is not None:
            query = query.filter(Scan.created_at >= filters['start_date'])
            
        if 'end_date' in filters and filters['end_date'] is not None:
            query = query.filter(Scan.created_at <= filters['end_date'])
        
        return query.count()

    @staticmethod
    def delete_scan(db: Session, job_id: str) -> bool:
        """
        Delete a scan by job_id
        """
        try:
            scan = db.query(Scan).filter(Scan.job_id == job_id).first()
            if scan:
                db.delete(scan)
                db.commit()
                return True
            return False
        except Exception as e:
            db.rollback()
            raise e
