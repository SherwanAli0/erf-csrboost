/* Sherwan Ali. Pure helpers shared by the workbench: escaping and number, duration and file name formatting. */
(function (root) {
  'use strict';
  const ENTITIES = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;', '`': '&#96;' };
  /* Text direction controls (U+202A to U+202E, U+2066 to U+2069, U+200E, U+200F, U+061C) can reorder the text around them,
     so they are never passed on: each one becomes a visible marker such as <U+202E>. */
  const BIDI_RANGES = [[0x202A, 0x202E], [0x2066, 0x2069], [0x200E, 0x200E], [0x200F, 0x200F], [0x061C, 0x061C]];
  const BIDI = new RegExp('[' + BIDI_RANGES.map(([a, b]) => String.fromCharCode(a) + (b > a ? '-' + String.fromCharCode(b) : '')).join('') + ']', 'g');
  const markBidi = value => String(value == null ? '' : value).replace(BIDI, c => `<U+${c.charCodeAt(0).toString(16).toUpperCase().padStart(4, '0')}>`);
  /* Every value that comes from a data file, a report or a server message goes through esc() before it is placed in HTML. */
  const esc = value => markBidi(value).replace(/[&<>"'`]/g, c => ENTITIES[c]);
  const UNAVAILABLE = 'Unavailable';
  function num(value, digits = 3) {
    if (value == null || typeof value !== 'number' || !Number.isFinite(value)) return UNAVAILABLE;
    return value.toLocaleString('en-US', { maximumFractionDigits: digits, minimumFractionDigits: 0 });
  }
  function fixed(value, digits = 3) {
    if (value == null || typeof value !== 'number' || !Number.isFinite(value)) return UNAVAILABLE;
    return value.toFixed(digits);
  }
  function signed(value, digits = 3) {
    if (value == null || typeof value !== 'number' || !Number.isFinite(value)) return UNAVAILABLE;
    const text = Math.abs(value).toFixed(digits);
    if (Number(text) === 0) return (0).toFixed(digits);
    return (value > 0 ? '+' : '-') + text;
  }
  function pct(value, digits = 1) {
    if (value == null || typeof value !== 'number' || !Number.isFinite(value)) return UNAVAILABLE;
    return `${(value * 100).toFixed(digits)}%`;
  }
  function int(value) {
    if (value == null || typeof value !== 'number' || !Number.isFinite(value)) return UNAVAILABLE;
    return Math.round(value).toLocaleString('en-US');
  }
  function seconds(value) {
    if (value == null || typeof value !== 'number' || !Number.isFinite(value)) return UNAVAILABLE;
    if (value < 10) return `${value.toFixed(2)} s`;
    if (value < 100) return `${value.toFixed(1)} s`;
    return `${Math.round(value)} s`;
  }
  function megabytes(bytes) {
    if (bytes == null || typeof bytes !== 'number' || !Number.isFinite(bytes)) return UNAVAILABLE;
    return `${(bytes / 1048576).toFixed(bytes < 10485760 ? 1 : 0)} MB`;
  }
  /* File sizes in sensible units: bytes, KB or MB. */
  function size(bytes) {
    if (bytes == null || typeof bytes !== 'number' || !Number.isFinite(bytes) || bytes < 0) return UNAVAILABLE;
    if (bytes < 1024) return `${Math.round(bytes)} ${Math.round(bytes) === 1 ? 'byte' : 'bytes'}`;
    if (bytes < 1048576) return `${(bytes / 1024).toFixed(bytes < 10240 ? 1 : 0)} KB`;
    return `${(bytes / 1048576).toFixed(bytes < 10485760 ? 1 : 0)} MB`;
  }
  function duration(secondsValue) {
    if (typeof secondsValue !== 'number' || !Number.isFinite(secondsValue) || secondsValue < 0) return UNAVAILABLE;
    if (secondsValue < 90) return `${Math.round(secondsValue)} seconds`;
    const minutes = Math.round(secondsValue / 60);
    if (minutes < 90) return `${minutes} minutes`;
    const hours = Math.round(secondsValue / 360) / 10;
    return `${hours} hours`;
  }
  /* The name part of a download file: lower case letters, digits and single hyphens, never empty. */
  function slug(text, fallback = 'analysis', max = 32) {
    const base = String(text == null ? '' : text).replace(/\.[A-Za-z0-9]{1,6}$/, '').toLowerCase()
      .replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, max).replace(/-+$/g, '');
    return base || fallback;
  }
  const EXPORT_FILES = {
    report_html: { suffix: 'report', extension: 'html', type: 'text/html' },
    report_json: { suffix: 'report', extension: 'json', type: 'application/json' },
    config_json: { suffix: 'config', extension: 'json', type: 'application/json' },
    repro_bundle: { suffix: 'repro-bundle', extension: 'zip', type: 'application/zip' }
  };
  function downloadName(kind, datasetName, dateIso) {
    const spec = EXPORT_FILES[kind];
    if (!spec) throw Error('Unknown download kind.');
    const day = /^\d{4}-\d{2}-\d{2}/.test(dateIso || '') ? dateIso.slice(0, 10) : 'undated';
    return `fyd-${slug(datasetName)}-${day}-${spec.suffix}.${spec.extension}`;
  }
  function plural(count, one, many) { return `${int(count)} ${count === 1 ? one : (many || one + 's')}`; }
  /* Join a list of text pieces into HTML-safe prose: "a, b and c". */
  function listText(items) {
    const safe = items.map(esc);
    if (safe.length <= 1) return safe.join('');
    return `${safe.slice(0, -1).join(', ')} and ${safe[safe.length - 1]}`;
  }
  const api = { esc, markBidi, num, fixed, signed, pct, int, seconds, megabytes, size, duration, slug, downloadName, plural, listText, EXPORT_FILES, UNAVAILABLE };
  root.FYD = root.FYD || {};
  root.FYD.util = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(globalThis);
