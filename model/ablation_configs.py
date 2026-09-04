from typing import Any, Dict


ABLATION_CONFIGS: Dict[int, Dict[str, Any]] = {
    1: {
        'name': 'X3D',
        'use_attention': False,
        'use_aspp': False,
        'use_transformer': False,
    },
    2: {
        'name': 'X3D + Attention',
        'use_attention': True,
        'use_aspp': False,
        'use_transformer': False,
    },
    3: {
        'name': 'X3D + ASPP',
        'use_attention': False,
        'use_aspp': True,
        'use_transformer': False,
    },
    4: {
        'name': 'X3D + Transformer',
        'use_attention': False,
        'use_aspp': False,
        'use_transformer': True,
    },
    5: {
        'name': 'X3D + Attention + ASPP',
        'use_attention': True,
        'use_aspp': True,
        'use_transformer': False,
    },
    6: {
        'name': 'X3D + ASPP + Transformer',
        'use_attention': False,
        'use_aspp': True,
        'use_transformer': True,
    },
    7: {
        'name': 'Full',
        'use_attention': True,
        'use_aspp': True,
        'use_transformer': True,
    },
    8: {
        'name': 'X3D + Attention + Transformer',
        'use_attention': True,
        'use_aspp': False,
        'use_transformer': True,
    },
}
