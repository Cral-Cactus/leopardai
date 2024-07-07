from pydantic import BaseModel
from typing import Optional

from .common import Metadata
from .deployment_operator_v1alpha1.job import leopardJobUserSpec, leopardJobStatus
from .deployment_operator_v1alpha1.job import *  # noqa: F401, F403


class leopardJob(BaseModel):
    metadata: Metadata
    spec: leopardJobUserSpec = leopardJobUserSpec()
    status: Optional[leopardJobStatus] = None