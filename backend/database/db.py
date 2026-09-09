"""
JointGuard Database Setup
Uses SQLAlchemy with SQLite for local prototype demo (swappable to PostgreSQL).
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from backend.config import settings

# For SQLite, enable check_same_thread=False for multi-threaded FastAPI execution
connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

from sqlalchemy import text

def get_db():
    """Dependency for API routes to get a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize DB tables and run column migrations if needed."""
    Base.metadata.create_all(bind=engine)
    if settings.DATABASE_URL.startswith("sqlite"):
        with engine.connect() as conn:
            try:
                conn.execute(text("ALTER TABLE sensor_readings ADD COLUMN motor_speed INTEGER"))
                conn.commit()
            except Exception:
                pass
            try:
                conn.execute(text("ALTER TABLE sensor_readings ADD COLUMN motor_running BOOLEAN"))
                conn.commit()
            except Exception:
                pass

