# todo
from typing import Union
from .api_resource import APIResourse
from .types.deployment import leopardDeployment
from .types.deployment_operator_v1alpha1.ingress import leopardIngressEndpoint
from .types.ingress import leopardIngress


class IngressAPI(APIResourse):
    def _to_name(self, name_or_ingress: Union[str, leopardIngress]) -> str:
        return (
            name_or_ingress
            if isinstance(name_or_ingress, str)
            else name_or_ingress.metadata.id_
        )

    def list_all(self):
        response = self._get("/ingress")
        return self.ensure_list(response, leopardIngress)

    def create(self, spec: leopardIngress):
        """
        Create an ingress with the given Ingress spec.
        """
        response = self._post("/ingress", json=self.safe_json(spec))
        return self.ensure_ok(response)

    def get(self, name_or_ingress: Union[str, leopardIngress]) -> leopardIngress:
        response = self._get(f"/ingress/{self._to_name(name_or_ingress)}")
        return self.ensure_type(response, leopardIngress)

    def delete(self, name_or_ingress: Union[str, leopardIngress]) -> bool:
        response = self._delete(f"/ingress/{self._to_name(name_or_ingress)}")
        return self.ensure_ok(response)

    def update(
        self, name_or_ingress: Union[str, leopardIngress], spec: leopardIngress
    ) -> Union[leopardIngress, str]:
        response = self._patch(
            f"/ingress/{self._to_name(name_or_ingress)}", json=self.safe_json(spec)
        )
        return self.ensure_type(response, leopardIngress)

    def create_endpoint(
        self, name_or_ingress: Union[str, leopardIngress], spec: leopardIngressEndpoint
    ) -> leopardIngress:
        """
        Create a ingressEndpoint with the given leopardIngress IngressEndpoint spec.
        """
        response = self._post(
            f"/ingress/{self._to_name(name_or_ingress)}/endpoint/deployment",
            json=self.safe_json(spec),
        )
        return self.ensure_type(response, leopardIngress)

    def delete_endpoint(
        self,
        name_or_ingress: Union[str, leopardIngress],
        name_or_deployment: Union[str, leopardDeployment, leopardIngressEndpoint],
    ) -> leopardIngress:
        """
        Deletes an endpoint for a given ingress and deployment.

        Args:
            name_or_ingress (Union[str, leopardIngress]): The ingress name or an instance of leopardIngress.
            dname_or_deployment_or_ingressendpoint (Union[str, leopardDeployment, leopardIngressEndpoint]):
                The deployment name,
                a string, or an instance of leopardDeployment, or an instance of leopardIngressEndpoint.

        Returns:
            leopardIngress: The response from the deletion request, as a leopardIngress object.

        This method converts the provided ingress and deployment to their respective names,
        constructs the appropriate URL, sends a DELETE request, and returns the response.
        """

        name = self._to_name(name_or_ingress)
        deployment_name = self._to_name(name_or_deployment)
        url = f"/ingress/{name}/endpoint/deployment/{deployment_name}"

        response = self.delete(url)
        return self.ensure_type(response, leopardIngress)