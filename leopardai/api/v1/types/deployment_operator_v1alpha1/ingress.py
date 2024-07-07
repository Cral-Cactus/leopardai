from enum import Enum
from pydantic import BaseModel
from typing import Optional, List


class LeastRequestLoadBalancer(BaseModel):
    choice_count: Optional[int] = None


class LoadBalanceConfig(BaseModel):
    least_request: Optional[LeastRequestLoadBalancer] = None


class leopardIngressEndpoint(BaseModel):
    deployment: Optional[str] = None
    weight: Optional[int] = None
    load_balance_config: Optional[LoadBalanceConfig] = None


class WorkspaceTierRateLimiter(BaseModel):
    pass


class RateLimitConfig(BaseModel):
    workspace_tier_ratelimiter: Optional[WorkspaceTierRateLimiter] = None


class leopardWorkspaceTokenAuth(BaseModel):
    remove_authorization_header: Optional[bool] = None


class AuthConfig(BaseModel):
    leopard_workspace_token_auth: Optional[leopardWorkspaceTokenAuth] = None


class leopardIngressLocality(BaseModel):
    region: Optional[str] = None


class TrafficShadowingConfig(BaseModel):
    percentage: Optional[int] = None
    endpoint: Optional[leopardIngressEndpoint] = None


class leopardIngressUserSpec(BaseModel):
    """
    The user spec of a leopard Ingress.
    """

    domain_name: str
    endpoints: Optional[List[leopardIngressEndpoint]] = None
    retelimit_config: Optional[RateLimitConfig] = None
    auth_config: Optional[AuthConfig] = None
    locality: Optional[leopardIngressLocality] = None
    traffic_shadowing_config: Optional[TrafficShadowingConfig] = None


class CustomDomainValidationStatus(str, Enum):
    Pending = "pending"
    Active = "active"
    Failed = "failed"