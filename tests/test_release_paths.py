"""Regression checks for moved checkouts and media paths with updated hashes."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]

class ReleasePaths(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT.parent)
        assert Path(self.tmp.name).resolve().is_relative_to(ROOT.parent.resolve())
        self.checkout = Path(self.tmp.name) / 'moved-checkout'
        shutil.copytree(ROOT, self.checkout, copy_function=os.link,
                        ignore=shutil.ignore_patterns('.git', 'models', 'paper', '__pycache__'))

    def tearDown(self):
        assert Path(self.tmp.name).resolve().is_relative_to(ROOT.parent.resolve())
        self.tmp.cleanup()

    def verify(self):
        return subprocess.run([sys.executable, '-X', 'utf8', str(self.checkout/'scripts/verify_assets.py')],
                              cwd=self.tmp.name, capture_output=True, text=True)

    def corrupt_path(self, replacement):
        file = self.checkout/'datasets/train.parquet'
        table = pq.read_table(file)
        column = table.schema.get_field_index('audios')
        values = table['audios'].to_pylist()
        index = next(i for i, refs in enumerate(values) if refs)
        values[index][0] = replacement(values[index][0])
        table = table.set_column(column, table.schema.field(column),
                                 pa.array(values, type=table.schema.field(column).type))
        # Atomic replacement avoids modifying hard-linked original files.
        new_file = file.with_suffix('.new.parquet')
        pq.write_table(table, new_file)
        os.replace(new_file, file)
        manifest_file = self.checkout/'datasets/asset_manifest.json'
        manifest = json.loads(manifest_file.read_text(encoding='utf8'))
        manifest['datasets/train.parquet']['sha256'] = hashlib.sha256(file.read_bytes()).hexdigest()
        new_manifest = manifest_file.with_suffix('.new.json')
        new_manifest.write_text(json.dumps(manifest), encoding='utf8')
        os.replace(new_manifest, manifest_file)

    def test_moved_checkout_from_unrelated_working_directory(self):
        result = self.verify()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['loader_resolved_datasets']['train']['media_references'], 296)

    def test_duplicated_dataset_prefix_fails_even_with_matching_hash(self):
        self.corrupt_path(lambda ref: 'datasets/' + ref)
        result = self.verify()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('FileNotFoundError', result.stderr)
        self.assertIn('train.parquet row 80 audios', result.stderr)

    def test_missing_media_fails_even_with_matching_hash(self):
        self.corrupt_path(lambda ref: 'media/nonexistent.wav')
        result = self.verify()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('media/nonexistent.wav', result.stderr)

if __name__ == '__main__':
    unittest.main()
