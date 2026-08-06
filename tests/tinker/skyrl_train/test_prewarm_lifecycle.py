"""Regression coverage for retaining service-owned inference after model unload."""

from unittest.mock import patch

import pytest

backend_module = pytest.importorskip("skyrl.backends.skyrl_train_backend")
worker_module = pytest.importorskip("skyrl.backends.skyrl_train.workers.worker")


def test_delete_model_keeps_prewarmed_inference():
    backend = object.__new__(backend_module.SkyRLTrainBackend)

    class Dispatch:
        stopped = False

        def shutdown(self):
            self.stopped = True

    dispatch = Dispatch()
    backend._model_ids_to_role = {"model": "policy"}
    backend._model_metadata = {"model": object()}
    backend._keep_inference_warm = True
    backend._dispatch = dispatch
    backend._cfg = object()
    backend._renderer = None

    class PlacementGroup:
        pg = object()

    placement_group = PlacementGroup()
    backend._colocate_pg = placement_group
    backend._base_lora_signature = (8, 16)
    backend._inference_engine_client = object()
    backend._inference_engines_initialized = True
    backend._server_groups = []
    backend._inference_router = None

    with patch.object(backend_module.ray.util, "remove_placement_group") as remove_placement_group:
        backend.delete_model("model")

    assert dispatch.stopped
    remove_placement_group.assert_called_once_with(placement_group.pg)
    assert backend._model_ids_to_role == {}
    assert backend._inference_engines_initialized
    assert backend._inference_engine_client is not None


def test_shutdown_releases_internal_worker_placement_group():
    group = object.__new__(worker_module.PPORayActorGroup)
    actor = object()
    placement_group = object()
    group._actor_handlers = [actor]
    group.actor_infos = [object()]
    group._internal_pg = placement_group

    with (
        patch.object(worker_module.ray, "kill") as kill,
        patch.object(worker_module.ray.util, "remove_placement_group") as remove_placement_group,
    ):
        group.shutdown()

    kill.assert_called_once_with(actor, no_restart=True)
    remove_placement_group.assert_called_once_with(placement_group)
    assert group._internal_pg is None
    assert group._actor_handlers == []
    assert group.actor_infos == []
