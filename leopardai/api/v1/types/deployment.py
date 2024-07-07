from pydantic import BaseModel
from typing import Optional

from .common import Metadata

from .deployment_operator_v1alpha1.deployment import (
    leopardDeploymentUserSpec,
    leopardDeploymentStatus,
)

# Implementation note: because users do need to use the deployment specs' detailed
# classes, we import them all here.
from .deployment_operator_v1alpha1.deployment import *  # noqa: F401, F403


class leopardDeployment(BaseModel):
    metadata: Optional[Metadata] = None
    spec: Optional[leopardDeploymentUserSpec] = None
    status: Optional[leopardDeploymentStatus] = None