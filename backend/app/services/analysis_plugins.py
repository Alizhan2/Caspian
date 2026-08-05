from abc import ABC, abstractmethod
from typing import Any


class AnalysisPlugin(ABC):
    """Common contract for future environmental monitoring modes."""
    mode: str

    @abstractmethod
    async def search_data(self, area: list[float], time_range: tuple[str, str]) -> dict[str, Any]: ...

    @abstractmethod
    def preprocess(self, scene: Any) -> Any: ...

    @abstractmethod
    def predict(self, image: Any) -> Any: ...

    @abstractmethod
    def postprocess(self, prediction: Any) -> Any: ...

    @abstractmethod
    def create_report(self, result: Any) -> Any: ...


class OilSpillPlugin(AnalysisPlugin):
    mode = "oil_spill"

    async def search_data(self, area: list[float], time_range: tuple[str, str]) -> dict[str, Any]:
        raise NotImplementedError("Use SceneCatalog in the orchestration service")

    def preprocess(self, scene: Any) -> Any:
        return scene

    def predict(self, image: Any) -> Any:
        return image

    def postprocess(self, prediction: Any) -> Any:
        return prediction

    def create_report(self, result: Any) -> Any:
        return result
