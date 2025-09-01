"""
Electrode group definitions for SFN2 sensor ROI ERP plotting.

Copied (minimally) from SFN/code/utils.py to avoid cross-package coupling.
"""

ELECTRODE_GROUPS = {
    "N1": {
        "bilateral": {
            "electrodes": [
                'E66', 'E65', 'E59', 'E60', 'E67', 'E71', 'E70',
                'E84', 'E76', 'E77', 'E85', 'E91', 'E90', 'E83'
            ],
            "description": "Bilateral N1 Regions"
        }
    },
    "P1": {
        "Oz": {
            "electrodes": ['E71', 'E75', 'E76', 'E70', 'E83', 'E74', 'E81', 'E82'],
            "description": "Parieto-Occipital Region"
        }
    },
    "P3b": {
        "midline": {
            "electrodes": ['E62', 'E78', 'E77', 'E72', 'E67', 'E61', 'E54', 'E55', 'E79'],
            "description": "Centro-Parietal Region"
        }
    }
}


