"""Tests for Megatron backend correctness fixes.

Tests that require megatron-core (GPU dependency) are skipped when it is not
installed.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import pytest


# NOTE: this duplicates the config helper in test_megatron_correctness.py, but that is
# intentional to keep the two tests independent.
def _fft_dispatch_cfg(weight_sync_backend: str = "nccl") -> SimpleNamespace:
    """Build the minimal ``self.cfg`` view that ``save_weights_for_sampler``
    inspects on the non-colocated path. Defaults to FFT (lora.rank=0) so
    the pause/resume branch is taken.

    ``weight_sync_backend`` defaults to ``"nccl"`` so the caller-pauses branch is
    exercised; pass ``"delta"`` for the branch where the sender pauses internally.
    """
    return SimpleNamespace(
        trainer=SimpleNamespace(
            strategy="fsdp",
            policy=SimpleNamespace(
                model=SimpleNamespace(lora=SimpleNamespace(rank=0)),
                megatron_config=SimpleNamespace(lora_config=SimpleNamespace(merge_lora=False)),
            ),
        ),
        generator=SimpleNamespace(
            inference_engine=SimpleNamespace(weight_sync_backend=weight_sync_backend, offload_kv_for_weight_sync=False),
        ),
    )


class TestSaveWeights:
    """Tests for `WorkerDispatch.save_weights_for_sampler`"""

    @pytest.mark.asyncio
    async def test_non_colocated_calls_pause_and_resume(self):
        from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch

        dispatch = WorkerDispatch.__new__(WorkerDispatch)
        dispatch.colocate_all = False
        dispatch.cfg = _fft_dispatch_cfg()
        dispatch._inference_engine_client = AsyncMock()
        dispatch._broadcast_to_inference_engines = MagicMock()
        dispatch._prepare_for_weight_sync = AsyncMock()
        dispatch._finish_weight_sync = MagicMock()
        dispatch.ensure_active_adapter = MagicMock()

        await dispatch.save_weights_for_sampler()

        dispatch._inference_engine_client.pause_generation.assert_awaited_once()
        dispatch._broadcast_to_inference_engines.assert_called_once()
        dispatch._inference_engine_client.resume_generation.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_non_colocated_delta_does_not_pause(self):
        """Delta sync owns pause/resume itself.

        ``DeltaWeightTransferSender._apply_receiver_update`` fetches before pausing and
        pauses only around the final reload, so the dispatcher must not pause as well --
        doing so would hold generation down across the whole publish+upload+fetch window
        instead of just the reload.
        """
        from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch

        dispatch = WorkerDispatch.__new__(WorkerDispatch)
        dispatch.colocate_all = False
        dispatch.cfg = _fft_dispatch_cfg(weight_sync_backend="delta")
        dispatch._inference_engine_client = AsyncMock()
        dispatch._broadcast_to_inference_engines = MagicMock()
        dispatch._prepare_for_weight_sync = AsyncMock()
        dispatch._finish_weight_sync = MagicMock()
        dispatch.ensure_active_adapter = MagicMock()

        await dispatch.save_weights_for_sampler()

        dispatch._inference_engine_client.pause_generation.assert_not_awaited()
        dispatch._inference_engine_client.resume_generation.assert_not_awaited()
        # The sync itself must still happen, and still be finalized.
        dispatch._broadcast_to_inference_engines.assert_called_once()
        dispatch._finish_weight_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_colocated_uses_wake_up(self):
        from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch

        dispatch = WorkerDispatch.__new__(WorkerDispatch)
        dispatch.colocate_all = True
        dispatch.cfg = _fft_dispatch_cfg()
        dispatch._inference_engine_client = AsyncMock()
        dispatch._broadcast_to_inference_engines = MagicMock()
        dispatch._prepare_for_weight_sync = AsyncMock()
        dispatch._finish_weight_sync = MagicMock()
        dispatch.ensure_active_adapter = MagicMock()

        await dispatch.save_weights_for_sampler()

        dispatch._prepare_for_weight_sync.assert_awaited_once_with(adapter_only_sync=False)
        dispatch._inference_engine_client.wake_up.assert_awaited()
        dispatch._inference_engine_client.pause_generation.assert_not_awaited()
        dispatch._inference_engine_client.resume_generation.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_colocated_pause_before_broadcast(self):
        from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch

        call_order = []

        dispatch = WorkerDispatch.__new__(WorkerDispatch)
        dispatch.colocate_all = False
        dispatch.cfg = _fft_dispatch_cfg()
        dispatch._inference_engine_client = AsyncMock()
        dispatch._inference_engine_client.pause_generation = AsyncMock(side_effect=lambda: call_order.append("pause"))
        dispatch._inference_engine_client.resume_generation = AsyncMock(side_effect=lambda: call_order.append("resume"))
        dispatch._broadcast_to_inference_engines = MagicMock(
            side_effect=lambda *args, **kwargs: call_order.append("broadcast")
        )
        dispatch._prepare_for_weight_sync = AsyncMock()
        dispatch._finish_weight_sync = MagicMock()
        dispatch.ensure_active_adapter = MagicMock()

        await dispatch.save_weights_for_sampler()

        assert call_order == ["pause", "broadcast", "resume"]

    @pytest.mark.asyncio
    async def test_non_colocated_resumes_on_broadcast_failure(self):
        """resume_generation must be called even if broadcast raises."""
        from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch

        dispatch = WorkerDispatch.__new__(WorkerDispatch)
        dispatch.colocate_all = False
        dispatch.cfg = _fft_dispatch_cfg()
        dispatch._inference_engine_client = AsyncMock()
        dispatch._broadcast_to_inference_engines = MagicMock(side_effect=RuntimeError("broadcast failed"))
        dispatch._prepare_for_weight_sync = AsyncMock()
        dispatch._finish_weight_sync = MagicMock()
        dispatch.ensure_active_adapter = MagicMock()

        with pytest.raises(RuntimeError, match="broadcast failed"):
            await dispatch.save_weights_for_sampler()

        dispatch._inference_engine_client.pause_generation.assert_awaited_once()
        dispatch._inference_engine_client.resume_generation.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_non_colocated_inplace_lora_skips_pause_and_resume(self):
        """In-place LoRA (lora.rank>0, no merge_lora) must NOT pause/resume.

        Mirrors the multi-tenant branch in
        ``save_weights_for_sampler``: when the engine's LoRA tensors are
        swapped in place via ``load_lora_adapter``, the weight sync is
        dispatched without any pause — load_lora_adapter is the engine-
        side primitive that's expected to be safe under in-flight
        requests on its own.
        """
        from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch

        cfg = _fft_dispatch_cfg()
        cfg.trainer.policy.model.lora.rank = 32  # in-place LoRA path
        cfg.trainer.policy.megatron_config.lora_config.merge_lora = False

        dispatch = WorkerDispatch.__new__(WorkerDispatch)
        dispatch.colocate_all = False
        dispatch.cfg = cfg
        dispatch._inference_engine_client = AsyncMock()
        dispatch._broadcast_to_inference_engines = MagicMock()
        dispatch._prepare_for_weight_sync = AsyncMock()
        dispatch._finish_weight_sync = MagicMock()
        dispatch.ensure_active_adapter = MagicMock()

        await dispatch.save_weights_for_sampler(model_id="lora-target")

        dispatch._broadcast_to_inference_engines.assert_called_once()
        dispatch._inference_engine_client.pause_generation.assert_not_awaited()
        dispatch._inference_engine_client.resume_generation.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_colocated_megatron_merge_lora_still_pauses(self):
        """Megatron + merge_lora keeps the pause/resume path (LoRA merged
        into the base weights → tensors flow over NCCL, not load_lora_adapter)."""
        from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch

        cfg = _fft_dispatch_cfg()
        cfg.trainer.strategy = "megatron"
        cfg.trainer.policy.model.lora.rank = 32
        cfg.trainer.policy.megatron_config.lora_config.merge_lora = True

        dispatch = WorkerDispatch.__new__(WorkerDispatch)
        dispatch.colocate_all = False
        dispatch.cfg = cfg
        dispatch._inference_engine_client = AsyncMock()
        dispatch._broadcast_to_inference_engines = MagicMock()
        dispatch._prepare_for_weight_sync = AsyncMock()
        dispatch._finish_weight_sync = MagicMock()
        dispatch.ensure_active_adapter = MagicMock()

        await dispatch.save_weights_for_sampler()

        dispatch._inference_engine_client.pause_generation.assert_awaited_once()
        dispatch._inference_engine_client.resume_generation.assert_awaited_once()


def _megatron_lora_cfg(*, merge_lora: bool = False, offload_after_step: bool = True) -> SimpleNamespace:
    """Megatron + LoRA policy config: trainable adapters and frozen base weights offload independently."""
    cfg = _fft_dispatch_cfg()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.policy.model.lora.rank = 32
    cfg.trainer.policy.megatron_config.lora_config.merge_lora = merge_lora
    cfg.trainer.policy.optimizer_config = SimpleNamespace(offload_after_step=offload_after_step)
    return cfg


def _state(model_on_gpu: bool, optimizer_on_gpu: bool, trainable_on_gpu: bool | None = None):
    """GPUState stand-in; ``trainable_on_gpu`` defaults to ``model_on_gpu`` (no split)."""
    if trainable_on_gpu is None:
        trainable_on_gpu = model_on_gpu
    return SimpleNamespace(
        model_on_gpu=model_on_gpu, optimizer_on_gpu=optimizer_on_gpu, trainable_on_gpu=trainable_on_gpu
    )


def _adapter_sync_dispatch(*, model_on_gpu: bool, optimizer_on_gpu: bool = False, trainable_on_gpu: bool | None = None):
    """Dispatch wired for the colocated adapter-only sync path
    (megatron + lora.rank>0 + merge_lora=False + colocate_all)."""
    from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch

    dispatch = WorkerDispatch.__new__(WorkerDispatch)
    dispatch.colocate_all = True
    dispatch.cfg = _megatron_lora_cfg()
    dispatch._inference_engine_client = AsyncMock()
    dispatch._inference_engine_client.increment_weight_version = MagicMock()
    dispatch._broadcast_to_inference_engines = MagicMock()
    dispatch._ensure_on_gpu = MagicMock()
    dispatch._offload = MagicMock()
    dispatch.empty_cache = MagicMock()
    dispatch._gpu_state = {"policy": _state(model_on_gpu, optimizer_on_gpu, trainable_on_gpu)}
    group = MagicMock()
    group.async_run_ray_method.return_value = "swap-future"
    dispatch._actor_groups = {"policy": group}
    return dispatch


class _FakeActorGroup:
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def backload_to_gpu(self, backload_optimizer=True, backload_model=True, model_scope="all"):
        self.calls.append(("backload", self.name, backload_optimizer, backload_model, model_scope))

    def offload_to_cpu(self, offload_optimizer=True, offload_model=True, model_scope="all"):
        self.calls.append(("offload", self.name, offload_optimizer, offload_model, model_scope))


def _fake_dispatch(cfg, initial_state):
    """Dispatch with real residency bookkeeping over recording fake actor groups.

    ``initial_state`` maps model name to ``(model_on_gpu, optimizer_on_gpu[, trainable_on_gpu])``.
    """
    from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch

    calls = []
    dispatch = WorkerDispatch.__new__(WorkerDispatch)
    dispatch.colocate_all = True
    dispatch.colocate_policy_ref = False
    dispatch.cfg = cfg
    dispatch._inference_engine_client = AsyncMock()
    dispatch._inference_engine_client.is_sleeping.return_value = False
    dispatch.empty_cache = MagicMock()
    dispatch._gpu_state = {name: _state(*flags) for name, flags in initial_state.items()}
    dispatch._actor_groups = {name: _FakeActorGroup(name, calls) for name in initial_state}
    return dispatch, calls


def _prepare_sync_dispatch(initial_state, *, offload_after_step=True):
    cfg = _fft_dispatch_cfg()
    cfg.trainer.policy.optimizer_config = SimpleNamespace(offload_after_step=offload_after_step)
    return _fake_dispatch(cfg, initial_state)


def _gpu_state_snapshot(dispatch):
    return {
        name: (state.model_on_gpu, state.optimizer_on_gpu, state.trainable_on_gpu)
        for name, state in dispatch._gpu_state.items()
    }


TRAINABLE_BACKLOAD = ("policy", {"need_optimizer": False, "need_model": False, "need_trainable": True})
ADAPTER_OFFLOAD = ("policy", {"offload_optimizer": False, "offload_model": True, "model_scope": "trainable"})


class TestAdapterOnlyColocatedSync:
    """Tests for adapter-only colocated LoRA sync.

    The export reads the LoRA DDP buffers, so the sync backloads only the
    trainable parameters (the adapters), never the TB-scale frozen masters,
    and offloads the adapters again before the engines generate.
    """

    @pytest.mark.asyncio
    async def test_cold_sync_backloads_only_adapters(self, monkeypatch):
        """Fully offloaded trainer: no engine sleep, no master backload; adapters in, then out."""
        from skyrl.backends.skyrl_train.workers import worker_dispatch as wd

        dispatch = _adapter_sync_dispatch(model_on_gpu=False)
        monkeypatch.setattr(wd.ray, "get", lambda _: None)

        await dispatch.save_weights_for_sampler(model_id="m1")

        dispatch._inference_engine_client.sleep.assert_not_awaited()
        assert dispatch._ensure_on_gpu.call_args_list[0] == call(TRAINABLE_BACKLOAD[0], **TRAINABLE_BACKLOAD[1])
        assert all(not c.kwargs.get("need_model") for c in dispatch._ensure_on_gpu.call_args_list)
        dispatch._offload.assert_called_once_with(ADAPTER_OFFLOAD[0], **ADAPTER_OFFLOAD[1])
        wake_tags = [c.kwargs["tags"] for c in dispatch._inference_engine_client.wake_up.await_args_list]
        assert wake_tags == [["weights"], ["kv_cache"]]
        dispatch._broadcast_to_inference_engines.assert_called_once_with(
            dispatch._inference_engine_client, model_id="m1"
        )
        dispatch._inference_engine_client.increment_weight_version.assert_called_once()

    @pytest.mark.asyncio
    async def test_adapters_offloaded_before_kv_cache_wake(self, monkeypatch):
        """The adapters leave the GPU before the engines take back their KV cache."""
        from skyrl.backends.skyrl_train.workers import worker_dispatch as wd

        dispatch = _adapter_sync_dispatch(model_on_gpu=False)
        monkeypatch.setattr(wd.ray, "get", lambda _: None)
        order = []
        dispatch._offload.side_effect = lambda *a, **k: order.append(("offload", k.get("model_scope")))
        dispatch._broadcast_to_inference_engines.side_effect = lambda *a, **k: order.append(("broadcast", None))
        dispatch._inference_engine_client.wake_up.side_effect = lambda tags: order.append(("wake", tags[0]))

        await dispatch.save_weights_for_sampler(model_id="m1")

        assert order == [("wake", "weights"), ("broadcast", None), ("offload", "trainable"), ("wake", "kv_cache")]

    @pytest.mark.asyncio
    async def test_swap_requires_only_trainable_resident(self, monkeypatch):
        """The adapter swap copies DDP param buffers, so it needs the adapters but not the base model."""
        from skyrl.backends.skyrl_train.workers import worker_dispatch as wd

        dispatch = _adapter_sync_dispatch(model_on_gpu=False)
        monkeypatch.setattr(wd.ray, "get", lambda _: None)

        await dispatch.save_weights_for_sampler(model_id="m1")

        assert dispatch._ensure_on_gpu.call_args_list, "swap must ensure adapter residency"
        assert all(not c.kwargs.get("need_model") for c in dispatch._ensure_on_gpu.call_args_list)
        assert all(c.kwargs.get("need_trainable") for c in dispatch._ensure_on_gpu.call_args_list)
        dispatch._actor_groups["policy"].async_run_ray_method.assert_called_once_with(
            "pass_through", "swap_to_adapter", "m1"
        )

    @pytest.mark.asyncio
    async def test_hot_sync_offloads_masters_and_optimizer_keeps_adapters(self, monkeypatch):
        """Post-optim sync: frozen masters and optimizer go, adapters stay for the export."""
        from skyrl.backends.skyrl_train.workers import worker_dispatch as wd

        dispatch = _adapter_sync_dispatch(model_on_gpu=True, optimizer_on_gpu=True)
        monkeypatch.setattr(wd.ray, "get", lambda _: None)

        await dispatch.save_weights_for_sampler(model_id="m1")

        assert dispatch._offload.call_args_list == [
            (("policy",), {"offload_optimizer": True, "offload_model": True, "model_scope": "frozen"}),
            ((ADAPTER_OFFLOAD[0],), ADAPTER_OFFLOAD[1]),
        ]
        dispatch.empty_cache.assert_called_once_with("policy")
        wake_tags = [c.kwargs["tags"] for c in dispatch._inference_engine_client.wake_up.await_args_list]
        assert wake_tags == [["weights"], ["kv_cache"]]
        dispatch._broadcast_to_inference_engines.assert_called_once_with(
            dispatch._inference_engine_client, model_id="m1"
        )


@pytest.mark.parametrize(
    ("initial_state", "expected_calls"),
    [
        # Cold: only the adapters come back.
        ({"policy": (False, False, False)}, [("backload", "policy", False, True, "trainable")]),
        # Adapters already resident from a previous sync: nothing moves.
        ({"policy": (False, False, True)}, []),
        # Hot after optim_step: masters + optimizer out, adapters stay.
        ({"policy": (True, True, True)}, [("offload", "policy", True, True, "frozen")]),
        # Model resident, optimizer already offloaded (offload_after_step).
        ({"policy": (True, False, True)}, [("offload", "policy", False, True, "frozen")]),
        # Another colocated model is offloaded whole.
        (
            {"policy": (False, False, False), "critic": (True, True, True)},
            [("offload", "critic", True, True, "all"), ("backload", "policy", False, True, "trainable")],
        ),
    ],
)
@pytest.mark.asyncio
async def test_prepare_for_adapter_only_sync_leaves_only_adapters_on_gpu(initial_state, expected_calls):
    dispatch, calls = _fake_dispatch(_megatron_lora_cfg(), initial_state)

    await dispatch._prepare_for_weight_sync(adapter_only_sync=True)

    expected_state = {name: (False, False, False) for name in initial_state}
    expected_state["policy"] = (False, False, True)
    assert _gpu_state_snapshot(dispatch) == expected_state
    assert calls == expected_calls
    # adapter only sync doesn't issue any sleep/ wake up calls to the inference engine
    dispatch._inference_engine_client.sleep.assert_not_awaited()
    dispatch._inference_engine_client.wake_up.assert_not_awaited()
    dispatch.empty_cache.assert_called_once_with("policy")


def test_finish_adapter_only_sync_offloads_adapters():
    dispatch, calls = _fake_dispatch(_megatron_lora_cfg(), {"policy": (False, False, True)})

    dispatch._finish_weight_sync(adapter_only_sync=True)

    assert calls == [("offload", "policy", False, True, "trainable")]
    assert _gpu_state_snapshot(dispatch) == {"policy": (False, False, False)}


def test_finish_full_sync_offloads_whole_lora_model():
    """merge_lora=True still backloads the whole policy for the merge; afterwards nothing stays."""
    dispatch, calls = _fake_dispatch(_megatron_lora_cfg(merge_lora=True), {"policy": (True, False, True)})

    dispatch._finish_weight_sync(adapter_only_sync=False)

    assert calls == [("offload", "policy", True, True, "all")]
    assert _gpu_state_snapshot(dispatch) == {"policy": (False, False, False)}


@pytest.mark.parametrize(
    ("offload_after_step", "initial_state", "expected_state", "expected_calls"),
    [
        (
            True,
            {"policy": (False, False)},
            {"policy": (True, False, True)},
            [("backload", "policy", False, True, "all")],
        ),
        (
            True,
            {"policy": (True, True)},
            {"policy": (True, False, True)},
            [("offload", "policy", True, False, "all")],
        ),
        (
            False,
            {"policy": (True, True)},
            {"policy": (True, True, True)},
            [],
        ),
        (
            True,
            {"policy": (False, False), "critic": (True, True)},
            {"policy": (True, False, True), "critic": (False, False, False)},
            [("offload", "critic", True, True, "all"), ("backload", "policy", False, True, "all")],
        ),
    ],
)
@pytest.mark.asyncio
async def test_prepare_for_full_sync_leaves_policy_weights_on_gpu(
    offload_after_step, initial_state, expected_state, expected_calls
):
    dispatch, calls = _prepare_sync_dispatch(initial_state, offload_after_step=offload_after_step)

    await dispatch._prepare_for_weight_sync(adapter_only_sync=False)

    assert _gpu_state_snapshot(dispatch) == expected_state
    assert calls == expected_calls
    dispatch._inference_engine_client.sleep.assert_awaited_once()
    dispatch._inference_engine_client.wake_up.assert_not_awaited()
    dispatch.empty_cache.assert_called_once_with("policy")


@pytest.mark.parametrize("offload_after_step", [False, True])
@pytest.mark.asyncio
async def test_weight_sync_honors_optimizer_offload_policy(offload_after_step):
    from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch

    cfg = _fft_dispatch_cfg()
    cfg.trainer.policy.optimizer_config = SimpleNamespace(offload_after_step=offload_after_step)

    dispatch = WorkerDispatch.__new__(WorkerDispatch)
    dispatch.colocate_all = True
    dispatch.cfg = cfg
    dispatch._inference_engine_client = AsyncMock()
    dispatch._inference_engine_client.is_sleeping.return_value = False
    dispatch.empty_cache = MagicMock()

    dispatch._gpu_state = {"policy": _state(True, True)}
    dispatch._ensure_on_gpu = MagicMock()
    dispatch._offload = MagicMock()

    await dispatch._prepare_for_weight_sync()

    dispatch._inference_engine_client.sleep.assert_awaited_once()
    dispatch._ensure_on_gpu.assert_called_once_with(
        "policy",
        need_optimizer=False,
        need_model=True,
    )
    if offload_after_step:
        dispatch._offload.assert_called_once_with("policy", offload_optimizer=True, offload_model=False)
    else:
        dispatch._offload.assert_not_called()

    dispatch._offload.reset_mock()
    dispatch._finish_weight_sync()
    dispatch._offload.assert_called_once_with("policy", offload_optimizer=offload_after_step, offload_model=True)


class TestSplitResidency:
    """``_ensure_on_gpu`` / ``_offload`` with a trainable/frozen split (Megatron LoRA)."""

    def test_need_trainable_cold_backloads_adapters_only(self):
        dispatch, calls = _fake_dispatch(_megatron_lora_cfg(), {"policy": (False, False, False)})

        dispatch._ensure_on_gpu("policy", need_optimizer=False, need_model=False, need_trainable=True)

        assert calls == [("backload", "policy", False, True, "trainable")]
        assert _gpu_state_snapshot(dispatch) == {"policy": (False, False, True)}

    def test_need_model_with_adapters_resident_backloads_frozen_only(self):
        dispatch, calls = _fake_dispatch(_megatron_lora_cfg(), {"policy": (False, False, True)})

        dispatch._ensure_on_gpu("policy", need_optimizer=True, need_model=True)

        assert calls == [("backload", "policy", True, True, "frozen")]
        assert _gpu_state_snapshot(dispatch) == {"policy": (True, True, True)}

    def test_need_model_cold_backloads_everything(self):
        dispatch, calls = _fake_dispatch(_megatron_lora_cfg(), {"policy": (False, False, False)})

        dispatch._ensure_on_gpu("policy", need_optimizer=False, need_model=True)

        assert calls == [("backload", "policy", False, True, "all")]
        assert _gpu_state_snapshot(dispatch) == {"policy": (True, False, True)}

    def test_need_trainable_without_split_backloads_whole_model(self):
        """FSDP (and full-parameter Megatron) move the model as one unit."""
        dispatch, calls = _prepare_sync_dispatch({"policy": (False, False)})

        dispatch._ensure_on_gpu("policy", need_optimizer=False, need_model=False, need_trainable=True)

        assert calls == [("backload", "policy", False, True, "all")]
        assert _gpu_state_snapshot(dispatch) == {"policy": (True, False, True)}

    def test_offload_frozen_keeps_adapters_resident(self):
        dispatch, calls = _fake_dispatch(_megatron_lora_cfg(), {"policy": (True, True, True)})

        dispatch._offload("policy", offload_optimizer=True, offload_model=True, model_scope="frozen")

        assert calls == [("offload", "policy", True, True, "frozen")]
        assert _gpu_state_snapshot(dispatch) == {"policy": (False, False, True)}

    def test_offload_trainable_clears_both_flags(self):
        dispatch, calls = _fake_dispatch(_megatron_lora_cfg(), {"policy": (False, False, True)})

        dispatch._offload("policy", offload_optimizer=False, offload_model=True, model_scope="trainable")

        assert calls == [("offload", "policy", False, True, "trainable")]
        assert _gpu_state_snapshot(dispatch) == {"policy": (False, False, False)}

    def test_offload_trainable_of_fully_resident_model_widens_to_all(self):
        """Adapters-only offload with the base weights resident would leave them untracked."""
        dispatch, calls = _fake_dispatch(_megatron_lora_cfg(), {"policy": (True, False, True)})

        dispatch._offload("policy", offload_optimizer=False, offload_model=True, model_scope="trainable")

        assert calls == [("offload", "policy", False, True, "all")]
        assert _gpu_state_snapshot(dispatch) == {"policy": (False, False, False)}

    def test_partial_scope_rejected_without_split(self):
        dispatch, _ = _prepare_sync_dispatch({"policy": (True, True)})

        with pytest.raises(ValueError, match="no trainable/frozen split"):
            dispatch._offload("policy", offload_model=True, model_scope="frozen")

    def test_offload_for_sampling_clears_adapter_only_residency(self):
        """Cold sample paths must not leave the adapters on the GPU either."""
        dispatch, calls = _fake_dispatch(_megatron_lora_cfg(), {"policy": (False, False, True)})

        dispatch.offload_for_sampling()

        assert calls == [("offload", "policy", True, True, "all")]
        assert _gpu_state_snapshot(dispatch) == {"policy": (False, False, False)}

    def test_inactive_model_with_adapters_resident_is_offloaded(self):
        """Adapter-only residency counts as resident when another model needs the GPU."""
        dispatch, calls = _fake_dispatch(
            _megatron_lora_cfg(), {"policy": (False, False, True), "critic": (False, False, False)}
        )

        dispatch._ensure_on_gpu("critic", need_optimizer=False, need_model=True)

        assert calls == [("offload", "policy", True, True, "all"), ("backload", "critic", False, True, "all")]
        assert _gpu_state_snapshot(dispatch) == {"policy": (False, False, False), "critic": (True, False, True)}


def test_offload_inactive_model_records_offloaded_state():
    """
    ``_offload_inactive_model`` performs a real ``offload_to_cpu()``, so it must
    record the model as not resident. Recording "resident" would make the next
    ``_ensure_on_gpu`` skip the backload and run the model from CPU.
    """
    from skyrl.backends.skyrl_train.workers.worker_dispatch import (
        GPUState,
        WorkerDispatch,
    )

    calls = []
    group = SimpleNamespace(offload_to_cpu=lambda *a, **k: calls.append("offload"))
    stub = SimpleNamespace(
        _actor_groups={"policy": group},
        _gpu_state={"policy": GPUState(model_on_gpu=True, optimizer_on_gpu=True, trainable_on_gpu=True)},
    )

    WorkerDispatch._offload_inactive_model(stub, "policy")

    assert calls == ["offload"]
    assert stub._gpu_state["policy"] == GPUState.offloaded()


def test_gpu_state_requires_explicit_intent():
    """GPUState must not be constructible without stating every field."""
    from skyrl.backends.skyrl_train.workers.worker_dispatch import GPUState

    with pytest.raises(TypeError):
        GPUState()
    with pytest.raises(TypeError):
        GPUState(model_on_gpu=True, optimizer_on_gpu=True)
