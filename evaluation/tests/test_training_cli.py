import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from rare_event_eval.cli import main


class TrainingCLI(unittest.TestCase):
    def test_fresh_training_provenance_and_aggregate_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root/'fictional.csv'
            data.write_text('signal,context,outcome\n' + ''.join(
                f'{i%17},{i%7},{"event" if i%4==0 else "clear"}\n' for i in range(120)), encoding='utf-8')
            output, html = root/'result.json', root/'result.html'
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = main(['train', str(data), '--target', 'outcome', '--positive-label', 'event',
                             '--negative-label', 'clear', '--model', 'logistic',
                             '--data-description', 'Fictional software test.', '--output', str(output), '--html', str(html)])
            self.assertEqual(code, 0)
            result = json.loads(output.read_text())
            row = result['outcomes']['logistic']
            self.assertEqual(row['status'], 'success')
            self.assertTrue(row['report']['data']['description'].startswith('Fictional'))
            for key in ('test_truth', 'test_scores', 'test_indices', 'test_groups'):
                self.assertNotIn(key, row)
            self.assertNotIn('cv', row['execution']['model_parameters'])
            self.assertIn('Fictional software test.', html.read_text())
