const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright");

async function main() {
  await fs.mkdir(".tmp/demo-qa", { recursive: true });
  const browser = await chromium.launch({ channel: "msedge", headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 980 } });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.location().url + " " + message.text());
  });
  const screenshot = (name) => page.screenshot({ path: `.tmp/demo-qa/${name}.png`, fullPage: true });
  async function download(name) {
    const button = page.getByTestId("stDownloadButton").filter({ hasText: name }).getByRole("button");
    const [file] = await Promise.all([page.waitForEvent("download"), button.click()]);
    return fs.readFile(await file.path(), "utf8");
  }
  try {
    await page.goto(process.env.WARNING_DEMO_URL || "http://127.0.0.1:8501");
    await page.getByRole("heading", { name: "高风险预警名单" }).waitFor();
    await page.getByTestId("stVegaLiteChart").first().locator("canvas, svg.marks").first().waitFor();
    await screenshot("01-demo");
    const csv = await download("下载预警名单");
    assert.match(csv, /rank,user_id,risk_score,risk_tier,risk_factors,suggested_action/);
    assert.match(csv, /DEMO-/);
    const sample = await download("下载示例用户表");

    await page.getByText("按风险档平均", { exact: true }).click();
    await page.getByRole("option", { name: "单个用户" }).click();
    await page.getByTestId("stSelectbox").filter({ hasText: "用户 ID" }).waitFor();
    await screenshot("02-single-user");

    await page.getByText("上传打分表", { exact: true }).click();
    const upload = page.locator('input[type="file"]');
    await upload.setInputFiles({ name: "example.csv", mimeType: "text/csv", buffer: Buffer.from(sample) });
    const scored = page.getByTestId("stMetric").filter({ hasText: "已打分" });
    await scored.getByText("10", { exact: true }).waitFor();
    await screenshot("03-upload");
    const invalid = sample.replace("DEMO-0001", "=1+1");
    await upload.setInputFiles({ name: "invalid.csv", mimeType: "text/csv", buffer: Buffer.from(invalid) });
    await page.getByText("无法打分：", { exact: false }).waitFor();
    await screenshot("04-invalid");

    await page.getByText("合成演示", { exact: true }).click();
    await scored.getByText("240", { exact: true }).waitFor();
    await page.setViewportSize({ width: 390, height: 844 });
    await page.getByRole("heading", { name: "高风险预警名单" }).scrollIntoViewIfNeeded();
    await screenshot("05-mobile");
    assert.deepEqual(errors, []);
    assert.equal(await page.getByTestId("stException").count(), 0);
    console.log("Browser check passed: charts, person view, downloads, upload, validation, mobile.");
  } finally {
    await browser.close();
  }
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
