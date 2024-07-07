from typing import Union, List, Iterator, Optional

from .api_resource import APIResourse
from .types.events import leopardEvent

from .types.job import leopardJob
from .types.replica import Replica


class JobAPI(APIResourse):
    def _to_name(self, name_or_job: Union[str, leopardJob]) -> str:
        return (  # type: ignore
            name_or_job if isinstance(name_or_job, str) else name_or_job.metadata.id_
        )

    def list_all(self) -> List[leopardJob]:
        responses = self._get("/jobs")
        return self.ensure_list(responses, leopardJob)

    def create(self, spec: leopardJob) -> leopardJob:
        """
        Run a photon with the given job spec.
        """
        response = self._post("/jobs", json=self.safe_json(spec))
        return self.ensure_type(response, leopardJob)

    def get(self, name_or_job: Union[str, leopardJob]) -> leopardJob:
        response = self._get(f"/jobs/{self._to_name(name_or_job)}")
        return self.ensure_type(response, leopardJob)

    def update(self, name_or_job: Union[str, leopardJob], spec: leopardJob) -> bool:
        raise NotImplementedError("Job update is not implemented yet.")

    def delete(self, name_or_job: Union[str, leopardJob]) -> bool:
        response = self._delete(f"/jobs/{self._to_name(name_or_job)}")
        return self.ensure_ok(response)

    def get_events(self, name_or_job: Union[str, leopardJob]) -> List[leopardEvent]:
        response = self._get(f"/jobs/{self._to_name(name_or_job)}/events")
        return self.ensure_list(response, leopardEvent)

    def get_replicas(self, name_or_job: Union[str, leopardJob]) -> List[Replica]:
        response = self._get(f"/jobs/{self._to_name(name_or_job)}/replicas")
        return self.ensure_list(response, Replica)

    def get_log(
        self,
        name_or_job: Union[str, leopardJob],
        replica: Union[str, Replica],
        timeout: Optional[int] = None,
    ) -> Iterator[str]:
        """
        Gets the log of the given job's specified replica. The log is streamed
        in chunks until timeout is reached. If timeout is not specified, the log will be
        streamed indefinitely, although you should not rely on this behavior as connections
        can be dropped when streamed for a long time.
        """
        replica_id = replica if isinstance(replica, str) else replica.metadata.id_
        response = self._get(
            f"/jobs/{self._to_name(name_or_job)}/replicas/{replica_id}/log",
            stream=True,
            timeout=timeout,
        )
        if not response.ok:
            raise RuntimeError(
                f"API call failed with status code {response.status_code}. Details:"
                f" {response.text}"
            )
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                yield chunk.decode("utf8")