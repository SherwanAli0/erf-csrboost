"""Runs the Node test scripts of the browser workbench (test_web_*.js) and checks the model list against the registry.

Node 18 or newer and no packages are needed. Without Node the tests are skipped, as in test_browser.py.
"""
import json
from pathlib import Path
import re
import shutil
import subprocess
import unittest

TESTS = Path(__file__).parent
WEB = TESTS.parent / 'src' / 'rare_event_eval' / 'web'


def node_major():
    node = shutil.which('node')
    if not node:
        return 0
    out = subprocess.run([node, '--version'], capture_output=True, text=True).stdout.strip()
    match = re.match(r'v(\d+)\.', out)
    return int(match.group(1)) if match else 0


NEEDS_NODE = unittest.skipUnless(node_major() >= 18, 'Node 18 or newer is required for the browser tests.')


@NEEDS_NODE
class WebNodeScripts(unittest.TestCase):
    def run_script(self, name):
        result = subprocess.run(['node', str(TESTS / name)], capture_output=True, text=True, cwd=str(TESTS), timeout=300)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        verdict = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(verdict['status'], 'passed', result.stdout + result.stderr)
        self.assertGreater(verdict['tests'], 0)
        return verdict

    def test_rendering_and_escaping(self):
        self.run_script('test_web_rendering.js')

    def test_analysis_flow_state_machine(self):
        self.run_script('test_web_flow.js')

    def test_api_client_and_tokens(self):
        self.run_script('test_web_api_client.js')

    def test_predictions_make_no_network_request(self):
        self.run_script('test_web_predictions.js')

    def test_static_scans_contrast_and_wiring(self):
        self.run_script('test_web_static.js')

    def test_page_survives_a_script_that_fails_to_load_and_a_failing_samples_request(self):
        self.run_script('test_web_resilience.js')

    def test_research_wording_matches_repository_documents(self):
        self.run_script('test_web_content.js')

    def test_integrated_tools_privacy_and_wiring(self):
        self.run_script('test_web_tools.js')

    def test_every_node_test_script_is_wired(self):
        scripts = sorted(path.name for path in TESTS.glob('test_web_*.js'))
        wired = sorted(set(re.findall(r"run_script\('(test_web_\w+\.js)'\)", Path(__file__).read_text(encoding='utf-8'))))
        self.assertEqual(scripts, wired)


@NEEDS_NODE
class ModelListMatchesRegistry(unittest.TestCase):
    def test_embedded_model_list_equals_the_package_registry(self):
        from rare_event_eval.registry import REGISTRY, package_model_ids
        script = ("const m=require(process.argv[1]);process.stdout.write(JSON.stringify(m));")
        result = subprocess.run(['node', '-e', script, str(WEB / 'models.js')], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        embedded = json.loads(result.stdout)
        expected = [{'id': spec.id, 'name': spec.display_name, 'recipe': spec.recipe_id, 'family': spec.family,
                     'cost': spec.cost_class, 'default': spec.default, 'description': spec.description}
                    for spec in (REGISTRY[model_id] for model_id in package_model_ids())]
        self.assertEqual(embedded, expected)

    def test_models_the_page_can_send_are_all_runnable(self):
        from rare_event_eval.training import MODEL_NAMES
        script = ("const m=require(process.argv[1]);process.stdout.write(JSON.stringify(m.map(x=>x.id)));")
        result = subprocess.run(['node', '-e', script, str(WEB / 'models.js')], capture_output=True, text=True, timeout=60)
        self.assertEqual(sorted(json.loads(result.stdout)), sorted(MODEL_NAMES))


class PackagingOfTheWebFolder(unittest.TestCase):
    def test_every_web_file_type_is_packaged(self):
        root = TESTS.parent
        pyproject = (root / 'pyproject.toml').read_text(encoding='utf-8')
        manifest = (root / 'MANIFEST.in').read_text(encoding='utf-8')
        for path in WEB.iterdir():
            if path.is_file() and not path.name.startswith('.'):
                self.assertIn(f'web/*{path.suffix}', pyproject, path.name)
                self.assertIn(f'*{path.suffix}', manifest.split('web', 1)[1], path.name)
        self.assertIn('web/*.json', pyproject, 'the recorded example is JSON')


if __name__ == '__main__':
    unittest.main()
