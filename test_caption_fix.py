"""
Quick test to verify caption fixes are working correctly
"""
import sys
from pathlib import Path

# Test the HTML template to verify figure numbering
sys.path.insert(0, str(Path(__file__).parent))

from code.utils.report_generator import HTML_TEMPLATE, SENSOR_ROI_ERP_SECTION_TEMPLATE

print("=" * 60)
print("TESTING CAPTION FIXES")
print("=" * 60)

# Check ROI ERP section template
print("\n1. ROI ERP Section Template:")
print("-" * 40)
if "Figure 0.1" in SENSOR_ROI_ERP_SECTION_TEMPLATE:
    print("[PASS] Figure 0.1 found (was Figure 0)")
else:
    print("[FAIL] Figure 0.1 not found")

if "Condition ERPs over canonical ROIs" not in SENSOR_ROI_ERP_SECTION_TEMPLATE:
    print("[PASS] Descriptive caption removed")
else:
    print("[FAIL] Descriptive caption still present")

# Check main sensor section template
print("\n2. Main Sensor Section Template:")
print("-" * 40)
if "Figure 1.1" in HTML_TEMPLATE:
    print("[PASS] Figure 1.1 found (was Figure 1)")
else:
    print("[FAIL] Figure 1.1 not found")

if "{sensor_main_caption_detail}" not in HTML_TEMPLATE:
    print("[PASS] Caption detail placeholder removed")
else:
    print("[FAIL] Caption detail placeholder still present")

# Check that we're using simple captions only
print("\n3. Caption Style Check:")
print("-" * 40)
# The template should just have the figure number, no descriptive text
import re
main_caption = re.search(r'<figcaption><strong>Figure 1\.1</strong>(.*?)</figcaption>', HTML_TEMPLATE)
if main_caption:
    caption_content = main_caption.group(1).strip()
    if caption_content == "":
        print("[PASS] Main sensor caption is clean (Figure 1.1 only)")
    else:
        print(f"[FAIL] Caption has extra content: {caption_content}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
