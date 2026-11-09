/* Sherwan Ali. Writes web/recorded-example.js from web/recorded-example.json, so that the recorded example also opens
   when index.html is opened from a file (a file: page cannot fetch a JSON file). Run it after the recorded example has
   been regenerated:  node evaluation/tests/sync_recorded_example.js
   test_web_content.js fails when the two files differ. The script changes nothing in the JSON. */
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const WEB = path.join(__dirname, '..', 'src', 'rare_event_eval', 'web');
const source = path.join(WEB, 'recorded-example.json'), target = path.join(WEB, 'recorded-example.js');
const data = JSON.parse(fs.readFileSync(source, 'utf8'));
const ascii = text => text.replace(/[\u007f-￿]/g, c => '\\u' + c.charCodeAt(0).toString(16).padStart(4, '0'));
const body = '/* Sherwan Ali. A derived copy of recorded-example.json (a recorded app run on a seeded synthetic table), so the example\n' +
  '   also opens from a file: URL. Regenerate with: node evaluation/tests/sync_recorded_example.js */\n' +
  'window.FYD_RECORDED_EXAMPLE = ' + ascii(JSON.stringify(data)) + ';\n';
fs.writeFileSync(target, body, 'utf8');
console.log(JSON.stringify({ written: 'recorded-example.js', bytes: Buffer.byteLength(body) }));
