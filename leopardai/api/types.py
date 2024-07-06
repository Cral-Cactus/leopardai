# flake8: noqa
"""
Types for the leopard AI API.

These types are used as wrappers of the json payloads used by the API.
"""

from enum import Enum
from typing import List, Optional, Union
import warnings
from pydantic import BaseModel, Field

from leopardai.config import leopard_RESERVED_ENV_NAMES, VALID_SHAPES

from .v1.types.common import Metadata

from .v1.types.deployment_operator_v1alpha1.deployment import (
    ResourceRequirement,
    TokenValue,
    TokenVar,
    EnvValue,
    EnvVar,
    MountOptions,
    Mount,
    ScaleDown,
    AutoScaler,
    HealthCheckLiveness,
    HealthCheck,
    leopardDeploymentState,
    ContainerPort,
    leopardContainer,
    AutoscalerCondition,
    AutoScalerStatus,
    leopardResourceAffinity,
)

from .v1.types.deployment import (
    leopardDeploymentUserSpec as DeploymentUserSpec,
    DeploymentEndpoint,
    leopardDeploymentStatus as DeploymentStatus,
    leopardDeployment as Deployment,
)

from .v1.types.job import leopardJob, leopardJobUserSpec, leopardJobState, leopardJobStatus