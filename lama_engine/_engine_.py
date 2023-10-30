import torch
from typing import Any, Optional, Tuple, Union
import ray
from ray.util.actor_pool import ActorPool
from ._actor_ import LaMaActor
import numpy as np
from typing import List


class LaMaEngine:
    def __init__(
        self,
        config_path: str,
        max_wh: int = 2560,
        actors_per_gpu: int = 1,
    ) -> None:
        self._config_path = config_path
        self._max_wh = max_wh
        self._actors_per_gpu = actors_per_gpu

        # start ray service
        ray.init(num_gpus=self.number_of_gpu)

        # create ray instance
        self._actors = [
            LaMaActor.options(num_gpus=self.gpu_resource_per_actor).remote(
                config_path=self.config_path
            )
            for _ in range(self.number_of_total_actor)
        ]
        self._actor_pool = ActorPool(self._actors)

    def __del__(self):
        # kill ray instance
        _ = [ray.kill(actor) for actor in self._actors]

        # shutdown ray service
        ray.shutdown(_exiting_interpreter=False)

    def run(
        self,
        inputs: List[Tuple[np.ndarray, np.ndarray]],
        device_id: Optional[int] = None,
    ) -> Union[List[np.ndarray], None]:
        """
        Inpainting with LaMa.

        Args:
            inputs (List[Tuple[np.ndarray, np.ndarray]]): tuple(Image[H x W x 3], Mask[H x W])로 이루어진 List를 입력으로 넣습니다.
            device_id (Optional[int], optional): None이 아니라면 해당 id의 actor로만 inference를 수행합니다. Defaults to None.

        Returns:
            Union[List[np.ndarray], None]: Result Image[H x W x 3]으로 이루어진 List를 반환합니다. 오류 발생 시 None을 반환합니다.
        """
        results = None
        try:
            if device_id != None:
                assert device_id > -1 and device_id < self.number_of_total_actor
                result_remote = [
                    self._actors[device_id].run.remote(image, mask)
                    for image, mask in inputs
                ]
                results = ray.get(result_remote)
            else:
                results = self._actor_pool.map(
                    lambda actor, v: actor.run.remote(v[0], v[1]), inputs
                )
                results = list(results)
        except Exception as ex:
            print(ex)
        finally:
            return results

    def __call__(
        self,
        inputs: List[Tuple[np.ndarray, np.ndarray]],
        device_id: Optional[int] = None,
    ) -> Union[List[np.ndarray], None]:
        return self.run(inputs=inputs, device_id=device_id)

    @property
    def config_path(self):
        return self._config_path

    @property
    def max_wh(self):
        return self._max_wh

    @property
    def actors_per_gpu(self):
        return self._actors_per_gpu

    @property
    def number_of_gpu(self):
        return torch.cuda.device_count()

    @property
    def number_of_total_actor(self):
        return self.actors_per_gpu * self.number_of_gpu

    @property
    def gpu_resource_per_actor(self):
        return self.number_of_gpu / self.number_of_gpu
