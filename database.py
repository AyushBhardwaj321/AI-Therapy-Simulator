# database.py
import datetime
from contextlib import contextmanager

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, ForeignKey, func, distinct
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from werkzeug.security import generate_password_hash, check_password_hash

DB_NAME = "therapy_sessions.db"
DATABASE_URL = f"sqlite:///{DB_NAME}"

# --- SQLAlchemy Setup ---
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- Context Manager for DB Sessions ---
@contextmanager
def get_db_session():
    """Provide a transactional scope around a series of operations."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- ORM Models ---
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False)  # 'admin' or 'therapist'
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    therapist_series = relationship("TherapySeries", back_populates="therapist", cascade="all, delete-orphan")

class TherapySeries(Base):
    __tablename__ = "therapy_series"
    id = Column(Integer, primary_key=True, index=True)
    therapist_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    patient_key = Column(String, nullable=False)
    llm_provider = Column(String, nullable=False)
    total_sessions = Column(Integer, nullable=True)
    session_duration_minutes = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    therapist = relationship("User", back_populates="therapist_series")
    messages = relationship("Message", back_populates="series", cascade="all, delete-orphan")
    evaluations = relationship("Evaluation", back_populates="series", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    series_id = Column(Integer, ForeignKey("therapy_series.id"), nullable=False)
    session_number = Column(Integer, nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    series = relationship("TherapySeries", back_populates="messages")

class Evaluation(Base):
    __tablename__ = "evaluations"
    id = Column(Integer, primary_key=True, index=True)
    series_id = Column(Integer, ForeignKey("therapy_series.id"), nullable=False)
    session_number = Column(Integer, nullable=False)
    report = Column(Text, nullable=False)
    actual_duration_seconds = Column(Integer, nullable=True)
    duration_deviation_reason = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    series = relationship("TherapySeries", back_populates="evaluations")

# --- Database Initialization ---
def init_db():
    """Initializes the database and creates tables if they don't exist."""
    Base.metadata.create_all(bind=engine)
    create_first_admin_if_not_exists()

# --- Public API Functions (Interface remains the same) ---

def create_first_admin_if_not_exists():
    """Creates a default admin user if no users exist."""
    with get_db_session() as db:
        if db.query(User).count() == 0:
            print("No users found. Creating default admin user (admin/admin123).")
            admin = User(
                username='admin',
                password_hash=generate_password_hash('admin123'),
                role='admin'
            )
            db.add(admin)
            db.commit()

def create_user(username, password, role):
    with get_db_session() as db:
        if db.query(User).filter(User.username == username).first():
            return False  # Username already exists
        new_user = User(
            username=username,
            password_hash=generate_password_hash(password),
            role=role
        )
        db.add(new_user)
        db.commit()
        return True

def get_user(username):
    with get_db_session() as db:
        user = db.query(User).filter(User.username == username, User.is_active == True).first()
        if user:
            return {c.name: getattr(user, c.name) for c in user.__table__.columns}
        return None

def get_all_therapists():
    with get_db_session() as db:
        therapists = db.query(User).filter(User.role == 'therapist').order_by(User.created_at.desc()).all()
        return [
            {"id": t.id, "username": t.username, "is_active": t.is_active, "created_at": t.created_at}
            for t in therapists
        ]

def update_user_password(user_id, new_password):
    with get_db_session() as db:
        user = db.get(User, user_id)
        if user:
            user.password_hash = generate_password_hash(new_password)
            db.commit()

def set_user_active_status(user_id, is_active):
    with get_db_session() as db:
        user = db.get(User, user_id)
        if user:
            user.is_active = is_active
            db.commit()

def start_new_therapy_series(therapist_id, patient_key, llm_provider):
    with get_db_session() as db:
        new_series = TherapySeries(
            therapist_id=therapist_id,
            patient_key=patient_key,
            llm_provider=llm_provider
        )
        db.add(new_series)
        db.commit()
        db.refresh(new_series)
        return new_series.id

def save_session_plan(series_id, total_sessions, duration_minutes):
    with get_db_session() as db:
        series = db.get(TherapySeries, series_id)
        if series:
            series.total_sessions = total_sessions
            series.session_duration_minutes = duration_minutes
            db.commit()

def save_message(series_id, session_number, role, content):
    with get_db_session() as db:
        new_message = Message(
            series_id=series_id,
            session_number=session_number,
            role=role,
            content=content
        )
        db.add(new_message)
        db.commit()

def save_evaluation(series_id, session_number, report, actual_duration_seconds):
    with get_db_session() as db:
        new_evaluation = Evaluation(
            series_id=series_id,
            session_number=session_number,
            report=report,
            actual_duration_seconds=actual_duration_seconds
        )
        db.add(new_evaluation)
        db.commit()

def save_duration_deviation_reason(series_id, session_number, reason):
    with get_db_session() as db:
        evaluation = db.query(Evaluation).filter_by(series_id=series_id, session_number=session_number).first()
        if evaluation:
            evaluation.duration_deviation_reason = reason
            db.commit()

def get_all_series(therapist_id=None):
    with get_db_session() as db:
        # Subquery to count completed sessions
        subq = db.query(
            Evaluation.series_id,
            func.count(distinct(Evaluation.session_number)).label('completed_sessions')
        ).group_by(Evaluation.series_id).subquery()

        query = db.query(TherapySeries, User.username, subq.c.completed_sessions) \
                  .join(User, TherapySeries.therapist_id == User.id) \
                  .outerjoin(subq, TherapySeries.id == subq.c.series_id)

        if therapist_id:
            query = query.filter(TherapySeries.therapist_id == therapist_id)
        
        results = query.order_by(TherapySeries.created_at.desc()).all()

        series_list = []
        for series, username, completed_count in results:
            series_dict = {c.name: getattr(series, c.name) for c in series.__table__.columns}
            series_dict['therapist_username'] = username
            series_dict['completed_sessions'] = completed_count or 0
            series_list.append(series_dict)
            
        return series_list

def get_series_details(series_id):
    with get_db_session() as db:
        series = db.get(TherapySeries, series_id)
        if series:
            return {c.name: getattr(series, c.name) for c in series.__table__.columns}
        return None

def get_messages_for_series(series_id):
    with get_db_session() as db:
        messages = db.query(Message).filter_by(series_id=series_id).order_by(Message.timestamp.asc()).all()
        return [
            {'role': m.role, 'content': m.content, 'session_number': m.session_number}
            for m in messages
        ]

def get_all_evaluations_for_series(series_id):
    with get_db_session() as db:
        evals = db.query(Evaluation).filter_by(series_id=series_id).order_by(Evaluation.session_number.asc()).all()
        return [{c.name: getattr(e, c.name) for c in e.__table__.columns} for e in evals]

def get_latest_session_number(series_id):
    with get_db_session() as db:
        result = db.query(func.max(Message.session_number)).filter(Message.series_id == series_id).scalar()
        return result or 0

