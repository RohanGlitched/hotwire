// Loads the live engine out of desktop/plugin.js (the one shipped copy) without
// importing the SDK: the engine function sits between two marker comments.
import { readFileSync, writeFileSync, mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, dirname } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))

export async function loadEngine() {
  const src = readFileSync(join(here, '..', 'desktop', 'plugin.js'), 'utf8')
  const a = src.indexOf('// ===== HOTWIRE LIVE ENGINE BEGIN =====')
  const b = src.indexOf('// ===== HOTWIRE LIVE ENGINE END =====')
  if (a < 0 || b < 0) throw new Error('engine markers not found in desktop/plugin.js')
  const dir = mkdtempSync(join(tmpdir(), 'hotwire-engine-'))
  const file = join(dir, 'engine.mjs')
  writeFileSync(file, src.slice(a, b) + '\nexport default hotwireLiveEngine()\n')
  return (await import(pathToFileURL(file).href)).default
}
