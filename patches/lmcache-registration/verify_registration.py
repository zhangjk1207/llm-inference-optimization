"""Focused regression tests for the deployed registration-liveness bug."""
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from lmcache.integration.vllm import vllm_multi_process_adapter as adapter
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    ContextEntry, LMCacheDrivenTransferModule,
)
from lmcache.v1.multiprocess.modules.management import ManagementModule
from lmcache.v1.multiprocess.protocol import RequestType, get_payload_classes
from lmcache.v1.multiprocess.transfer_context import LMCacheDrivenTransferContext
from lmcache.v1.multiprocess.transport.zmq_impl.client import ZmqMultiprocessClient
from lmcache.v1.multiprocess.transport.zmq_impl.server import get_zmq_handler_specs


class LivenessTests(unittest.TestCase):
    def module(self):
        module = object.__new__(LMCacheDrivenTransferModule)
        module._lock = threading.Lock()
        module._cache_contexts = {
            7: ContextEntry(Mock(), 'model', 4, last_seen=0),
        }
        module._release_entries = Mock()
        return module

    def test_ping_before_first_transfer_uses_startup_grace(self):
        module = self.module()
        with patch('lmcache.v1.multiprocess.modules.lmcache_driven_transfer.time.monotonic', return_value=1):
            self.assertTrue(module.touch_instance(7))
        with patch('lmcache.v1.multiprocess.modules.lmcache_driven_transfer.time.monotonic', return_value=46):
            self.assertEqual(module.reap_stale_instances(30, 90), [])
        with patch('lmcache.v1.multiprocess.modules.lmcache_driven_transfer.time.monotonic', return_value=92):
            self.assertEqual(module.reap_stale_instances(30, 90), [7])

    def test_idle_worker_survives_registration_grace_with_heartbeats(self):
        module = self.module()
        for now in (10, 50, 100, 200):
            with patch('lmcache.v1.multiprocess.modules.lmcache_driven_transfer.time.monotonic', return_value=now):
                self.assertTrue(module.touch_instance(7))
                self.assertEqual(module.reap_stale_instances(30, 90), [])

    def test_dead_ready_worker_is_reaped(self):
        module = self.module()
        with patch('lmcache.v1.multiprocess.modules.lmcache_driven_transfer.time.monotonic', return_value=1):
            module.touch_instance(7)
            module.get_and_touch_context_entry(7, transfer_activity=True)
        with patch('lmcache.v1.multiprocess.modules.lmcache_driven_transfer.time.monotonic', return_value=32):
            self.assertEqual(module.reap_stale_instances(30, 90), [7])

    def test_registered_ping_detects_missing_primary_context(self):
        module = self.module()
        management = ManagementModule(Mock(), liveness_targets=[module], registration_targets={'register_kv_cache': module})
        self.assertTrue(management.ping(None))
        self.assertTrue(management.ping_registered(7, 'register_kv_cache'))
        self.assertFalse(management.ping_registered(8, 'register_kv_cache'))
        self.assertFalse(management.ping_registered(7, 'register_q_cache'))


class HeartbeatTests(unittest.TestCase):
    def heartbeat(self):
        event = threading.Event(); event.set()
        thread = adapter.HeartbeatThread(Mock(), event, interval=1, instance_id=7, registration_type='register_kv_cache')
        callback = Mock(return_value=True)
        thread.register_recover_callback(callback)
        return thread, event, callback

    def test_missing_registration_recovers_even_while_server_healthy(self):
        thread, event, callback = self.heartbeat()
        callback.side_effect = lambda: (self.assertFalse(event.is_set()) or True)
        with patch.object(adapter, 'send_registered_ping', return_value=False):
            thread._execute()
        callback.assert_called_once()
        self.assertTrue(event.is_set())

    def test_failed_recovery_retries_next_cycle(self):
        thread, event, callback = self.heartbeat()
        callback.side_effect = [False, True]
        with patch.object(adapter, 'send_registered_ping', return_value=False):
            thread._execute(); self.assertFalse(event.is_set())
            thread._execute(); self.assertTrue(event.is_set())
        self.assertEqual(callback.call_count, 2)

    def test_outage_does_not_register_until_server_returns(self):
        thread, event, callback = self.heartbeat()
        with patch.object(adapter, 'send_registered_ping', return_value=None):
            thread._execute()
        callback.assert_not_called(); self.assertFalse(event.is_set())
        with patch.object(adapter, 'send_registered_ping', return_value=False):
            thread._execute()
        callback.assert_called_once(); self.assertTrue(event.is_set())

    def test_shutdown_does_not_recover_or_set_health(self):
        thread, event, callback = self.heartbeat()
        event.clear()
        with patch.object(adapter.HeartbeatThread, 'stop_requested', new_callable=lambda: property(lambda self: True)), patch.object(adapter, 'send_registered_ping', return_value=False):
            thread._execute()
        callback.assert_not_called(); self.assertFalse(event.is_set())


class RegistrationTests(unittest.TestCase):
    def worker(self):
        worker = object.__new__(adapter.LMCacheMPWorkerAdapter)
        worker._registration_lock = threading.RLock()
        worker._shutdown_started = False
        worker.transfer_ctx = None
        worker.instance_id = 7; worker.model_name = 'model'
        worker.parallel_strategy = SimpleNamespace(kv_world_size=4)
        worker.blocks_in_chunk = 1; worker.req_client = Mock()
        worker._mq_timeout = 1; worker._mp_transfer_mode = 'lmcache_driven'
        worker._layout_hints = None; worker.engine_group_infos = []
        worker._ensure_heartbeat_started = Mock()
        return worker

    def test_registration_ack_starts_heartbeat_before_first_request(self):
        worker = self.worker()
        context = Mock(spec=LMCacheDrivenTransferContext)
        with patch.object(adapter, 'create_transfer_context', return_value=context):
            worker.register_kv_caches({'layer': Mock()}, layout_hints=Mock())
        worker._ensure_heartbeat_started.assert_called_once()
        self.assertIs(worker.transfer_ctx, context)
        self.assertEqual(worker._primary_registration_type, 'register_kv_cache')

    def test_registration_failure_never_publishes_context_or_starts_heartbeat(self):
        worker = self.worker()
        context = Mock(spec=LMCacheDrivenTransferContext)
        context.register.side_effect = TimeoutError()
        with patch.object(adapter, 'create_transfer_context', return_value=context), self.assertRaises(ConnectionError):
            worker.register_kv_caches({'layer': Mock()}, layout_hints=Mock())
        self.assertIsNone(worker.transfer_ctx)
        worker._ensure_heartbeat_started.assert_not_called()
        context.close.assert_called_once()

    def test_recovery_reuses_ipc_context_and_preserves_inflight_resources(self):
        worker = self.worker()
        context = Mock(spec=LMCacheDrivenTransferContext); worker.transfer_ctx = context
        with patch.object(adapter, 'create_transfer_context') as factory:
            for _ in range(5): worker._send_register_kv_caches_request({'layer': Mock()})
        factory.assert_not_called(); context.close.assert_not_called()
        self.assertEqual(context.register.call_count, 5)
        self.assertIs(worker.transfer_ctx, context)

    def test_shutdown_prevents_registration(self):
        worker = self.worker(); worker._shutdown_started = True
        with patch.object(adapter, 'create_transfer_context') as factory, self.assertRaises(ConnectionError):
            worker._send_register_kv_caches_request({'layer': Mock()})
        factory.assert_not_called()

    def test_zmq_protocol_and_handler_are_wired_without_renumbering_ping(self):
        self.assertEqual(RequestType.PING.value, 22)
        self.assertEqual(RequestType.PING_REGISTERED.value, 34)
        self.assertEqual(get_payload_classes(RequestType.PING_REGISTERED), [int, str])
        management = ManagementModule(Mock())
        self.assertIn(RequestType.PING_REGISTERED, [s.request_type for s in get_zmq_handler_specs(management)])
        client = ZmqMultiprocessClient(Mock())
        with patch.object(client, '_call') as call:
            client.ping_registered(7, 'register_kv_cache')
        call.assert_called_once_with(RequestType.PING_REGISTERED, 7, 'register_kv_cache')


if __name__ == '__main__':
    unittest.main(verbosity=2)
