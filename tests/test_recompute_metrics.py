"""Check portable metric manifests and metadata from archived predictions."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('recompute', ROOT/'paper/recompute_metrics.py')
recompute = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recompute)

class RecomputeMetrics(unittest.TestCase):
    def test_windows_paths_and_escape_rejection(self):
        self.assertEqual(recompute.manifest_path('results\\HARPO\\metrics_step_38.json'),
                         recompute.manifest_path('results/HARPO/metrics_step_38.json'))
        for name in ['../outside.json', '/outside.json', 'C:/outside.json', 'results/../../outside.json']:
            with self.assertRaises(ValueError):
                recompute.manifest_path(name)

    def test_relocated_checkout_and_legacy_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            paper = Path(temp)/'checkout/paper'
            paper.mkdir(parents=True)
            shutil.copy2(ROOT/'paper/recompute_metrics.py', paper/'recompute_metrics.py')
            shutil.copytree(ROOT/'paper/results', paper/'results')
            manifest_path = paper/'results/source_manifest.json'
            manifest = json.loads(manifest_path.read_text(encoding='utf8'))
            self.assertTrue(all('\\' not in name for name in manifest))
            reference_path = paper/'results/reported_metrics.json'
            reference = json.loads(reference_path.read_text(encoding='utf8'))
            obsolete = ['independent_final_test_set', 'source_declared_independent_final_test_set',
                        'independence_status', 'source_sha256', 'source_row_line_range']
            self.assertTrue(all(name not in reference for name in obsolete))
            # Exercise historical reference metadata and Windows-exported keys.
            reference.update({name: 'obsolete' for name in obsolete})
            reference['source_kind'] = 'no original predictions JSON'
            reference_path.write_text(json.dumps(reference), encoding='utf8')
            for windows in [False, True]:
                keys = {name.replace('/', chr(92)) if windows else name: item
                        for name, item in manifest.items()}
                manifest_path.write_text(json.dumps(keys), encoding='utf8')
                process = subprocess.run([sys.executable, '-X', 'utf8', str(paper/'recompute_metrics.py')],
                                         cwd=temp, capture_output=True, text=True, encoding='utf8')
                self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
                output = json.loads((paper/'source_data/verified_data.json').read_text(encoding='utf8'))
                self.assertTrue(all(name not in output for name in obsolete))
                self.assertIn('archived full model outputs', output['source_kind'])
                self.assertEqual(len(output['source_files']), 2)
                for model in ['harpo', 'v4']:
                    self.assertEqual(output['models'][model]['accuracy'], reference['models'][model]['accuracy'])

if __name__ == '__main__':
    unittest.main()
