# models/retirement/retirement_scenario.py

from models import db
from sqlalchemy.orm import relationship, backref
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON
from sqlalchemy.sql import func


class RetirementScenario(db.Model):
    __tablename__ = "retirement_scenarios"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    scenario_name = Column(String(100), nullable=False)

    # JSON blob of every input field (baseline_params) for easy reload + compare
    inputs_json = Column(
        JSON,
        nullable=False,
        comment="Stores the full set of planner inputs as a JSON object"
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )

    user = relationship(
        "User",
        backref=backref("retirement_scenarios", lazy="dynamic")
    )

    def __repr__(self) -> str:
        return f"<RetirementScenario id={self.id} name={self.scenario_name}>"


