import assert from "node:assert/strict";
import { access, readFile, readdir } from "node:fs/promises";
import test from "node:test";

test("builds a GitHub Pages entry point with repository-relative assets", async () => {
  const html = await readFile(new URL("../dist-pages/index.html", import.meta.url), "utf8");

  assert.match(html, /<title>DerivativeLab — Options Pricing & Risk Workbench<\/title>/);
  assert.match(html, /\/Quantitative-research-Model\/assets\/index-[^\"]+\.js/);
  assert.match(html, /\/Quantitative-research-Model\/assets\/index-[^\"]+\.css/);
  assert.match(html, /\/Quantitative-research-Model\/favicon\.svg/);
});

test("ships the complete interactive analytics model", async () => {
  const source = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  const assets = await readdir(new URL("../dist-pages/assets/", import.meta.url));

  assert.match(source, /function blackScholes/);
  assert.match(source, /function binomial/);
  assert.match(source, /function monteCarlo/);
  assert.match(source, /function impliedVolatility/);
  assert.match(source, /function greeks/);
  assert.match(source, /function simulateHedging/);
  assert.match(source, /function generateChain/);
  assert.match(source, /Options chain/);
  assert.ok(assets.some((file) => file.endsWith(".js")));
  assert.ok(assets.some((file) => file.endsWith(".css")));
  await access(new URL("../dist-pages/og.png", import.meta.url));
});
