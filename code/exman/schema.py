from sqlmodel import Field, SQLModel, UniqueConstraint
from typing import Optional
from datetime import datetime


class Step(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    run_id: int = Field(index=True, foreign_key="run.id")
    episode_id: int = Field(index=True)
    iteration_id: int = Field(index=True)
    time: float
    state: str
    action: str
    running_objective: float
    discounted_value: float
    undiscounted_value: float


class Run(SQLModel, table=True):
    id: int = Field(primary_key=True)
    timestamp: datetime = Field(index=True)
    run_path: str
    name: str
    agent: str = Field(index=True)
    system: str = Field(default=None, index=True, foreign_key="names.system")
    seed: Optional[int] = Field(default=None)
    sampling_time: float
    overrides: Optional[str] = Field(default=None)
    hostname: str


class Names(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("system"),)

    id: int = Field(default=None, primary_key=True)
    system: str = Field(index=True)
    state: str
    action: str


class Value(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    run_id: int = Field(index=True, foreign_key="run.id")
    episode_id: int = Field(index=True)
    iteration_id: int = Field(index=True)
    discounted_value: float = Field(index=True)
    undiscounted_value: float = Field(index=True)
