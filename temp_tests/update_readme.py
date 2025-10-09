from pathlib import Path

readme = Path("README.md")
text = readme.read_text(encoding="utf-8")
marker_start = "### Full pipeline behavior and config pairing"
marker_end = "### Output locations"
if marker_start not in text or marker_end not in text:
    raise SystemExit("Markers not found")
start = text.index(marker_start)
end = text.index(marker_end)
new_section = """### Full pipeline behavior and config pairing\n\n-   Always pass a sensor-space YAML to the full pipeline.\n-   The full pipeline runs sensor analysis first and parses the sensor stats report.\n-   If at least one significant sensor cluster is found, the pipeline searches the sensor config directory for every YAML that (a) ends with the same contrast slug (for example `_cardinality1_vs_cardinality2.yaml`) and (b) declares `domain: \"source\"`.\n    -   Keep the sensor file named `sensor_<slug>.yaml`. Name each source method `source_<method>_<slug>.yaml` (e.g., `source_dspm_cardinality1_vs_cardinality2.yaml`, `source_loreta_cardinality1_vs_cardinality2.yaml`).\n    -   All discovered source configs are executed in alphabetical order, so you can run multiple inverse methods without touching the CLI command.\n-   If no companion source configs are found (or none produce significant clusters), the combined report still includes the full sensor results and notes that the corresponding source analysis was skipped.\n\n### Output locations\n"""
readme.write_text(text[:start] + new_section + text[end:], encoding="utf-8")

