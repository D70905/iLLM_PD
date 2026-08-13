import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname.replace(/^\/(.:)/, "$1")), "..");
const tableDir = path.join(root, "Source_Data", "_final_tables_0813");
const outPath = path.join(root, "Source_Data", "SourceData_iLLM-PD_NC_final.xlsx");

const sheets = [
  "README",
  "Supplementary_source_index",
  "Table1_LTPP_sections",
  "Table2_ablation",
  "Fig2_trajectory",
  "Fig3_LTPP_design_response",
  "Fig4_method_comparison",
  "Fig4_MEPDG_rutting",
  "Fig5_temperature_fatigue",
  "Fig6_OOD_response",
];

const workbook = Workbook.create();

for (const sheetName of sheets) {
  const csvPath = path.join(tableDir, `${sheetName}.csv`);
  const csvText = await fs.readFile(csvPath, "utf8");
  await workbook.fromCSV(csvText, { sheetName });
}

for (const sheetName of sheets) {
  const ws = workbook.worksheets.getItem(sheetName);
  ws.showGridLines = false;
  const used = ws.getUsedRange(true);
  if (!used) continue;

  const values = used.values;
  const rowCount = values.length;
  const colCount = values[0]?.length ?? 0;
  if (rowCount === 0 || colCount === 0) continue;

  const header = ws.getRangeByIndexes(0, 0, 1, colCount);
  header.format.fill.color = "#EAF1F8";
  header.format.font.bold = true;
  header.format.font.color = "#1F2937";
  header.format.wrapText = true;
  header.format.borders = {
    bottom: { style: "medium", color: "#9CA3AF" },
  };

  used.format.font.name = "Arial";
  used.format.font.size = 10;
  used.format.borders = {
    insideHorizontal: { style: "thin", color: "#E5E7EB" },
  };
  used.format.autofitColumns();
  used.format.autofitRows();
  ws.freezePanes.freezeRows(1);
}

const readme = workbook.worksheets.getItem("README");
readme.getRange("A1:B1").format.fill.color = "#D9EAF7";
readme.getRange("A1:B1").format.font.bold = true;
readme.getRange("A:B").format.columnWidth = 42;

const sourceIndex = workbook.worksheets.getItem("Supplementary_source_index");
sourceIndex.getRange("A:D").format.columnWidth = 34;

const errorScan = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(errorScan.ndjson);

await fs.mkdir(path.dirname(outPath), { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outPath);
console.log(`saved ${outPath}`);
