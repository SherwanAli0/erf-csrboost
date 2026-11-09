from pathlib import Path
import shutil
import subprocess
import unittest


class BrowserParity(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'), 'Node is required for browser-core parity checks.')
    def test_browser_metrics_match_python_fixtures(self):
        script=Path(__file__).with_name('test_browser_parity.js')
        result=subprocess.run(['node',str(script)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('"status":"passed"',result.stdout)
