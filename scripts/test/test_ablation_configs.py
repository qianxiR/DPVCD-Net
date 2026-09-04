import gc
import unittest
from argparse import Namespace

import torch

from model.ablation_configs import ABLATION_CONFIGS
from model.trainer import Trainer
from model.trainer_ablation import create_ablation_model


class AblationConfigTest(unittest.TestCase):
    """
    入参:
    - 七组统一消融配置及模型构造函数

    方法:
    - 验证模块开关、可选模块数量以及配置7与Full模型的一致性

    出参:
    - unittest测试结果
    """

    def setUp(self) -> None:
        """
        入参:
        - 无

        方法:
        - 创建不加载预训练权重的最小模型参数

        出参:
        - None
        """
        self.args = Namespace(
            num_perception_frame=1,
            pretrained='',
            in_height=64,
            in_width=64,
        )

    def test_required_configurations(self) -> None:
        """
        入参:
        - 集中的消融配置表

        方法:
        - 对照已确认的七组Attention、ASPP和Transformer开关组合

        出参:
        - 编号、名称和模块组合完全一致
        """
        expected = {
            1: ('X3D', False, False, False),
            2: ('X3D + Attention', True, False, False),
            3: ('X3D + ASPP', False, True, False),
            4: ('X3D + Transformer', False, False, True),
            5: ('X3D + Attention + ASPP', True, True, False),
            6: ('X3D + ASPP + Transformer', False, True, True),
            7: ('Full', True, True, True),
            8: ('X3D + Attention + Transformer', True, False, True),
        }
        actual = {
            config_id: (
                config['name'],
                config['use_attention'],
                config['use_aspp'],
                config['use_transformer'],
            )
            for config_id, config in ABLATION_CONFIGS.items()
        }
        self.assertEqual(actual, expected)

    def test_models_construct_only_enabled_modules(self) -> None:
        """
        入参:
        - 七组消融配置

        方法:
        - 逐组构造模型并检查禁用模块为空、启用模块具备四个尺度

        出参:
        - 每个模型的实际模块与配置表一致
        """
        for config_id, config in ABLATION_CONFIGS.items():
            with self.subTest(config_id=config_id):
                torch.manual_seed(16)
                model = create_ablation_model(self.args, config_id)
                self.assertEqual(
                    len(model.attention_modules),
                    4 if config['use_attention'] else 0,
                )
                self.assertEqual(
                    len(model.aspp_modules),
                    4 if config['use_aspp'] else 0,
                )
                self.assertEqual(
                    model.st_fusion is not None,
                    config['use_transformer'],
                )
                del model
                gc.collect()

    def test_config_seven_matches_full_model(self) -> None:
        """
        入参:
        - 相同随机种子和模型参数

        方法:
        - 分别构造Trainer与配置7，逐项比较状态字典键、形状和初始值

        出参:
        - 配置7与Full模型结构及初始化完全一致
        """
        torch.manual_seed(16)
        full_model = Trainer(self.args)
        torch.manual_seed(16)
        ablation_full = create_ablation_model(self.args, 7)

        full_state = full_model.state_dict()
        ablation_state = ablation_full.state_dict()
        self.assertEqual(tuple(full_state), tuple(ablation_state))
        for key, reference_value in full_state.items():
            self.assertTrue(
                torch.equal(reference_value, ablation_state[key]),
                msg=key,
            )


if __name__ == '__main__':
    unittest.main()
