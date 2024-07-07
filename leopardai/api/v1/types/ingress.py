from pydantic import BaseModel
from typing import Optional

from .common import Metadata
from .deployment_operator_v1alpha1.ingress import (
    leopardIngressUserSpec,
    CustomDomainValidationStatus,
)


class leopardIngressStatus(BaseModel):
    """
    The status of a leopard Ingress.
    """

    # Inlined v1alpha1.leopardIngressStatus
    validation_status: Optional[CustomDomainValidationStatus] = None
    message: Optional[str] = None
    # additional properties
    expected_cname_target: Optional[str] = None
    expected_dns01_channelge_target: Optional[str] = None


class leopardIngress(BaseModel):
    metadata: Metadata
    spec: leopardIngressUserSpec
    status: Optional[leopardIngressStatus] = None
    expected_cname_target: Optional[str] = None
    expected_dns01_channelge_target: Optional[str] = None