from sqlalchemy import Column, String, Integer, Float, DateTime, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import datetime
import uuid
import json

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    email = Column(String(100), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    password = Column(String(200), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    role = Column(String(20), default='user')
    
    # Relationship with scans
    scans = relationship("Scan", back_populates="user", cascade="all, delete-orphan")
    # Relationship with shares
    shared_links = relationship("Share", back_populates="creator", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(name='{self.name}', email='{self.email}')>"
    
    def to_dict(self):
        return {
            "name": self.name,
            "email": self.email,
            "created_at": self.created_at.isoformat(),
            "role": self.role
        }

class Scan(Base):
    __tablename__ = 'scans'
    
    id = Column(Integer, primary_key=True)
    job_id = Column(String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    file_name = Column(String(255), nullable=False)
    status = Column(String(20), default='processing')  # processing, completed, failed
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    metastasis_count = Column(Integer, nullable=True)
    total_volume = Column(Float, nullable=True)
    metastasis_volumes = Column(String(1000), nullable=True)  # Stored as JSON string
    
    # Relationship with user
    user = relationship("User", back_populates="scans")
    
    def __repr__(self):
        return f"<Scan(job_id='{self.job_id}', status='{self.status}')>"
    
    def to_dict(self):
        # Parse metastasis_volumes from JSON string to Python list if it exists
        metastasis_volumes = None
        if self.metastasis_volumes:
            try:
                metastasis_volumes = json.loads(self.metastasis_volumes)
            except json.JSONDecodeError:
                # If parsing fails, return it as is
                metastasis_volumes = self.metastasis_volumes
        
        return {
            "job_id": self.job_id,
            "file_name": self.file_name,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "metastasis_count": self.metastasis_count,
            "total_volume": self.total_volume,
            "metastasis_volumes": metastasis_volumes
        }

class Share(Base):
    __tablename__ = 'shares'
    
    id = Column(Integer, primary_key=True)
    share_token = Column(String(255), unique=True, nullable=False)
    job_id = Column(String(36), nullable=False)
    created_by = Column(Integer, ForeignKey('users.id'), nullable=False)
    expires_at = Column(Integer, nullable=False)  # Unix timestamp
    allow_download = Column(Boolean, default=False)
    include_detailed_analysis = Column(Boolean, default=False)
    created_at = Column(Integer, nullable=False)  # Unix timestamp
    is_revoked = Column(Boolean, default=False)
    
    # Relationship with user
    creator = relationship("User", back_populates="shared_links")
    
    def __repr__(self):
        return f"<Share(token='{self.share_token}', job_id='{self.job_id}')>"
    
    def to_dict(self):
        return {
            "share_token": self.share_token,
            "job_id": self.job_id,
            "created_by": self.created_by,
            "expires_at": self.expires_at,
            "allow_download": self.allow_download,
            "include_detailed_analysis": self.include_detailed_analysis,
            "created_at": self.created_at,
            "is_revoked": self.is_revoked
        }
