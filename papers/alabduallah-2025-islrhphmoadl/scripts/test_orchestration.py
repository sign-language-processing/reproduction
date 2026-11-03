"""Cloud handoff must never duplicate the existing GPU call."""
import ast
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock


class OrchestrationTests(unittest.TestCase):
    def controller(self, root):
        tree = ast.parse(Path(__file__).with_name('modal_app.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'orchestrate')
        node.decorator_list = []
        full, successor, outputs, modal = Mock(), Mock(), Mock(), Mock()
        modal.current_function_call_id.return_value = 'controller'
        successor.spawn.return_value.object_id = 'next-controller'
        scope = {'json': json, 'time': time, 'Path': lambda _: root,
                 'full': full, 'outputs': outputs, 'modal': modal}
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<orchestrate>', 'exec'), scope)
        function = scope['orchestrate']
        scope['orchestrate'] = successor
        return function, full, successor

    def test_handoff_waits_for_committed_completion_without_new_gpu(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'run').mkdir()
            (root / 'run/execution-ledger.json').write_text(json.dumps({'calls': [
                {'function_call_id': 'old', 'state': 'completed'}]}))
            function, full, successor = self.controller(root)
            result = function('run', 72, time.time()+7200, 'old')
            self.assertEqual(result['state'], 'successor_queued')
            full.spawn.assert_not_called()
            successor.spawn.assert_called_once()

    def test_finished_campaign_does_not_spawn_successor(self):
        with tempfile.TemporaryDirectory() as temp:
            function, full, successor = self.controller(Path(temp))
            full.spawn.return_value.object_id = 'gpu-call'
            full.spawn.return_value.get.return_value = {'complete': True}
            result = function('run', 72, time.time()+50000)
            self.assertEqual(result['state'], 'completed')
            successor.spawn.assert_not_called()

    def test_expired_handoff_stops_without_duplicate_gpu(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'run').mkdir()
            (root / 'run/execution-ledger.json').write_text(json.dumps({'calls': [
                {'function_call_id': 'old', 'state': 'reserved'}]}))
            function, full, successor = self.controller(root)
            with self.assertRaisesRegex(RuntimeError, 'no committed completion'):
                function('run', 72, time.time()-1, 'old')
            full.spawn.assert_not_called()
            successor.spawn.assert_not_called()
