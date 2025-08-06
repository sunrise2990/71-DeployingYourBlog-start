# models/retirement.py

from models import db
from sqlalchemy.orm import relationship, backref
from sqlalchemy.sql import func
from sqlalchemy import Integer, String, Column, DateTime, ForeignKey, JSON

class RetirementScenario(db.Model):
    __tablename__ = "retirement_scenarios"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    scenario_name = Column(String(100), nullable=False)
    inputs_json = Column(JSON, nullable=False)  # Store form inputs as JSON
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user = relationship("User", backref=backref("retirement_scenarios"))

