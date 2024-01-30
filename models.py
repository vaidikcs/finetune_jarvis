from contextlib import contextmanager
import sqlalchemy as sq
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.orm import sessionmaker
import datetime

Base = declarative_base()

Session = None


def create_session(db_conn_string):
    engine_global = sq.create_engine(db_conn_string, pool_size=20, max_overflow=10)
    global Session
    Session = sessionmaker(bind=engine_global)


@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""

    # engine = sq.create_engine(db_conn_string)
    # Session = sessionmaker(bind = engine)
    session = Session()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()


def currentGMTTimestamp():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


class FinetuneJobsDetails(Base):
    """
    SQLAlchemy model for finetune_jobs_details table.
    """
    __tablename__ = 'finetune_jobs_details'

    job_id = sq.Column(sq.Integer, primary_key=True, autoincrement=True)
    model_name = sq.Column(sq.String(60))
    train_dataset = sq.Column(sq.String(250))
    config = sq.Column(sq.String(256))
    org_id = sq.Column(sq.String(50))
    user_id = sq.Column(sq.String(50))
    status = sq.Column(sq.String(250))
    created_on = sq.Column(sq.DateTime)
    final_model_name = sq.Column(sq.String(45))

    def __init__(self, job_id, model_name, train_dataset, config, org_id, user_id, status, final_model_name=None):
        """
        Constructor for the FinetuneJobsDetails class.
        """
        self.job_id = job_id
        self.model_name = model_name
        self.train_dataset = train_dataset
        self.config = config
        self.org_id = org_id
        self.user_id = user_id
        self.status = status
        self.created_on = currentGMTTimestamp()
        self.final_model_name = final_model_name


def set_status(job_id, status):
    with session_scope() as db:
        job_data = db.query(FinetuneJobsDetails).filter_by(
            job_id=job_id).first()

        job_data.status = status
        db.commit()
        db.close()