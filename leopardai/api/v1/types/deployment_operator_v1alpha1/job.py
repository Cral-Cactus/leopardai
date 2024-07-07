from enum import Enum
from pydantic import BaseModel
from typing import List, Optional

from .affinity import leopardResourceAffinity
from .deployment import leopardContainer, leopardMetrics, EnvVar, Mount


class leopardJobUserSpec(BaseModel):
    """
    The desired state of a leopard Job.
    """

    resource_shape: Optional[str] = None
    affinity: Optional[leopardResourceAffinity] = None
    container: leopardContainer = leopardContainer()
    completions: int = 1
    parallelism: int = 1
    max_failure_retry: Optional[int] = None
    max_job_failure_retry: Optional[int] = None
    envs: List[EnvVar] = []
    mounts: List[Mount] = []
    image_pull_secrets: List[str] = []
    ttl_seconds_after_finished: Optional[int] = None
    intra_job_communication: Optional[bool] = None
    privileged: Optional[bool] = None
    metrics: Optional[leopardMetrics] = None


DefaultTTLSecondsAfterFinished: int = 600


class leopardJobState(str, Enum):
    Starting = "Starting"
    Running = "Running"
    Failed = "Failed"
    Completed = "Completed"
    Deleting = "Deleting"
    Restarting = "Restarting"
    Unknown = ""


class leopardJobStatusDetails(BaseModel):
    """
    The current status of a leopard Job.
    """

    job_name: Optional[str] = None
    state: leopardJobState
    ready: int
    active: int
    failed: int
    succeeded: int
    creation_time: Optional[int] = None
    completion_time: Optional[int] = None


class leopardJobStatus(leopardJobStatusDetails):
    job_history: List[leopardJobStatusDetails] = []