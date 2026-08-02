# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import re
import random
import numpy as np

# ==============================================================================
# 1. 集中管理所有采样配置
# ==============================================================================

SAMPLING_CONFIG_DICT = {
    "l10": {
        # ==============================================================================
        # L10: 全量保留 (Ground Truth)
        # 目的：作为 Ground Truth，保留原始数据的每一个细节。
        # 特征：不丢弃任何背景、不截断任何属性列表、保留所有摄影参数、坐标和关系。
        # ==============================================================================
        # 自然语言描述采样概率
        "description": {"ratio": 1.0},
        # 场景、氛围等属性的简单随机采样率
        "scene_properties": {
            "setting": 1.0,
            "atmosphere": 1.0,
            "lighting": 1.0,
            "style": {"ratio": 1.0, "sample_if_not_contains": ("photo", 1.0)},
        },
        # 摄影属性的采样配置
        "photography": {
            "layout": {"ratio": 1.0, "sample_if_not_contains": ("single", 1.0)},
            "shot_type": {"ratio": 1.0, "sample_if_not_contains": (None, 1.0)},
            "camera_angle": {"ratio": 1.0, "sample_if_not_contains": ("eye", 1.0)},
            "lens_and_effect": {"ratio": 1.0, "sample_if_not_contains": ("standard lens with deep focus", 1.0)},
        },
        # 元素内部摄影属性的采样率
        "element_photography": {
            "composition": {"ratio": 1.0, "sample_if_not_contains": ("full", 1.0)},
            "view_perspective": {"ratio": 1.0, "sample_if_not_contains": ("front", 1.0)},
            "camera_angle": {"ratio": 1.0, "sample_if_not_contains": ("eye", 1.0)},
            "visibility": {"ratio": 1.0, "sample_if_not_contains": (None, 1.0)},
        },
        # 列表采样的概率模型参数
        "list_sampling": {
            "attributes_actions": {"strength": 1.0, "decay": 0, "scale": 1},  # decay=0 表示列表不衰减，全选
            "property_ratio": 1.0, # 100% 概率采样属性
            "relationships": {"strength": 1.0, "decay": 0, "scale": 1},
            "drop_position_ratio": 0, # 保留所有坐标
            "drop_depth_ratio": 0,    # 保留所有深度
            "drop_relationship_ratio": 0, # 保留所有关系
        }
    },
    "l10_dropAllPD": {
        # ==============================================================================
        # L10: 全量保留 (Ground Truth)
        # 目的：作为 Ground Truth，保留原始数据的每一个细节。
        # 特征：不丢弃任何背景、不截断任何属性列表、保留所有摄影参数、坐标和关系。
        # ==============================================================================
        # 自然语言描述采样概率
        "description": {"ratio": 1.0},
        # 场景、氛围等属性的简单随机采样率
        "scene_properties": {
            "setting": 1.0,
            "atmosphere": 1.0,
            "lighting": 1.0,
            "style": {"ratio": 1.0, "sample_if_not_contains": ("photo", 1.0)},
        },
        # 摄影属性的采样配置
        "photography": {
            "layout": {"ratio": 1.0, "sample_if_not_contains": ("single", 1.0)},
            "shot_type": {"ratio": 1.0, "sample_if_not_contains": (None, 1.0)},
            "camera_angle": {"ratio": 1.0, "sample_if_not_contains": ("eye", 1.0)},
            "lens_and_effect": {"ratio": 1.0, "sample_if_not_contains": ("standard lens with deep focus", 1.0)},
        },
        # 元素内部摄影属性的采样率
        "element_photography": {
            "composition": {"ratio": 1.0, "sample_if_not_contains": ("full", 1.0)},
            "view_perspective": {"ratio": 1.0, "sample_if_not_contains": ("front", 1.0)},
            "camera_angle": {"ratio": 1.0, "sample_if_not_contains": ("eye", 1.0)},
            "visibility": {"ratio": 1.0, "sample_if_not_contains": (None, 1.0)},
        },
        # 列表采样的概率模型参数
        "list_sampling": {
            "attributes_actions": {"strength": 1.0, "decay": 0, "scale": 1},  # decay=0 表示列表不衰减，全选
            "property_ratio": 1.0, # 100% 概率采样属性
            "relationships": {"strength": 1.0, "decay": 0, "scale": 1},
            "drop_position_ratio": 0, # 保留所有坐标
            "drop_depth_ratio": 0,    # 保留所有深度
            "drop_relationship_ratio": 0, # 保留所有关系
            "drop_all_posistion_depth_ratio": 0.5, # 50%的概率丢弃坐标和深度
        }
    },
    "l9": {
        # ==============================================================================
        # L9: 高保真去噪 (High Fidelity Denoised)
        # 目的：去掉显而易见的默认值，模拟“极高质量的详细 Prompt”。
        # 特征：开始丢弃“标准镜头”、“平视视角”等默认参数，属性列表开始有轻微衰减。
        # ==============================================================================
        # 自然语言描述采样概率
        "description": {"ratio": 1.0},
        # 场景、氛围等属性的简单随机采样率
        "scene_properties": {
            "setting": 1.0,
            "atmosphere": 0.5, # 氛围词开始变得可选
            "lighting": 0.5,
            "style": {"ratio": 0.2, "sample_if_not_contains": ("photorealistic", 1.0)}, # 风格更小概率被采样，一般为photo
        },
        # 摄影属性的采样配置
        "photography": {
            "layout": {"ratio": 0.2, "sample_if_not_contains": ("single", 1.0)},
            "shot_type": {"ratio": 0.2, "sample_if_not_contains": (None, 1.0)},
            "camera_angle": {"ratio": 0.2, "sample_if_not_contains": ("eye", 1.0)},
            "lens_and_effect": {"ratio": 0.2, "sample_if_not_contains": ("standard lens with deep focus", 1.0)},
        },
        # 元素内部摄影属性的采样率
        "element_photography": {
            "composition": {"ratio": 0.2, "sample_if_not_contains": ("full", 1.0)}, # 构图更小概率被采样，一般为full，特殊的已经被merge到short_caption
            "view_perspective": {"ratio": 0.5, "sample_if_not_contains": ("front", 1.0)}, # 视角更小概率被采样，一般为front，特殊的已经被merge到short_caption
            "camera_angle": {"ratio": 0.2, "sample_if_not_contains": ("eye", 1.0)},
            "visibility": {"ratio": 0.2, "sample_if_not_contains": (None, 1.0)},
        },
        # 列表采样的概率模型参数
        "list_sampling": {
            "attributes_actions": {"strength": 1.0, "decay": 0.5, "scale": 1.0}, # 平均约保留79%
            "property_ratio": 0.8, # 80%的概率采样属性列表
            "relationships": {"strength": 1.0, "decay": 0.5, "scale": 1.0}, # 平均约保留79%
            "drop_position_ratio": 0,
            "drop_depth_ratio": 0,
            "drop_relationship_ratio": 0,
        }
    },
    "l8.5": {
        # ==============================================================================
        # L8: 丰富细节 (Rich Detail)
        # 目的：模拟“非常详细但非面面俱到”的描述。
        # 特征：背景元素开始少量丢失，坐标和深度信息不再是100%必备，属性列表衰减加快。
        # ==============================================================================
        # 自然语言描述采样概率
        "description": {"ratio": 1.0},
        # 场景、氛围等属性的简单随机采样率
        "scene_properties": {
            "setting": 1.0,
            "atmosphere": 0.5,
            "lighting": 0.5,
            "style": {"ratio": 0.2, "sample_if_not_contains": ("photorealistic", 1.0)},
        },
        # 摄影属性的采样配置
        "photography": {
            "layout": {"ratio": 0.2, "sample_if_not_contains": ("single image", 1.0)},
            "shot_type": {"ratio": 0.2, "sample_if_not_contains": (None, 1.0)},
            "camera_angle": {"ratio": 0.2, "sample_if_not_contains": ("eye", 1.0)},
            "lens_and_effect": {"ratio": 0.2, "sample_if_not_contains": ("standard lens with deep focus", 1.0)},
        },
        # 元素内部摄影属性的采样率
        "element_photography": {
            "composition": {"ratio": 0.2, "sample_if_not_contains": ("full", 1.0)},
            "view_perspective": {"ratio": 0.5, "sample_if_not_contains": ("front", 1.0)},
            "camera_angle": {"ratio": 0.2, "sample_if_not_contains": ("eye", 1.0)},
            "visibility": {"ratio": 0.2, "sample_if_not_contains": (None, 1.0)},
        },
        # 列表采样的概率模型参数
        "list_sampling": {
            "attributes_actions": {"strength": 1.0, "decay": 0.65, "scale": 1.0}, # 平均约保留74%
            "property_ratio": 0.75, # 75%的概率采样属性列表
            "relationships": {"strength": 1.0, "decay": 0.5, "scale": 1.0},
            "drop_background_element_ratio": 0.1, # 10%的背景元素被丢弃
            "drop_position_ratio": 0.2, # 20%的概率不带坐标
            "drop_depth_ratio": 0.2, # 20%的概率不带深度
            "drop_relationship_ratio": 0.25, # 25%的概率丢弃关系
        }
    },
    "l8": {
        # ==============================================================================
        # L8: 丰富细节 (Rich Detail)
        # 目的：模拟“非常详细但非面面俱到”的描述。
        # 特征：背景元素开始少量丢失，坐标和深度信息不再是100%必备，属性列表衰减加快。
        # ==============================================================================
        # 自然语言描述采样概率
        "description": {"ratio": 1.0},
        # 场景、氛围等属性的简单随机采样率
        "scene_properties": {
            "setting": 1.0,
            "atmosphere": 0.5,
            "lighting": 0.5,
            "style": {"ratio": 0.2, "sample_if_not_contains": ("photorealistic", 1.0)},
        },
        # 摄影属性的采样配置
        "photography": {
            "layout": {"ratio": 0.2, "sample_if_not_contains": ("single image", 1.0)},
            "shot_type": {"ratio": 0.2, "sample_if_not_contains": (None, 1.0)},
            "camera_angle": {"ratio": 0.2, "sample_if_not_contains": ("eye", 1.0)},
            "lens_and_effect": {"ratio": 0.2, "sample_if_not_contains": ("standard lens with deep focus", 1.0)},
        },
        # 元素内部摄影属性的采样率
        "element_photography": {
            "composition": {"ratio": 0.2, "sample_if_not_contains": ("full", 1.0)},
            "view_perspective": {"ratio": 0.5, "sample_if_not_contains": ("front", 1.0)},
            "camera_angle": {"ratio": 0.2, "sample_if_not_contains": ("eye", 1.0)},
            "visibility": {"ratio": 0.2, "sample_if_not_contains": (None, 1.0)},
        },
        # 列表采样的概率模型参数
        "list_sampling": {
            "attributes_actions": {"strength": 1.0, "decay": 0.65, "scale": 1.0}, # 平均约保留74%
            "property_ratio": 0.75, # 75%的概率采样属性列表
            "relationships": {"strength": 1.0, "decay": 0.5, "scale": 1.0},
            "drop_background_element_ratio": 0.3, # 30%的背景元素被丢弃
            "drop_position_ratio": 0.2, # 20%的概率不带坐标
            "drop_depth_ratio": 0.2, # 20%的概率不带深度
            "drop_relationship_ratio": 0.5, # 50%的概率丢弃关系
        }
    },
    "l8_dropBGFG": {
        # ==============================================================================
        # L8: 丰富细节 (Rich Detail)
        # 目的：模拟“非常详细但非面面俱到”的描述。
        # 特征：背景元素开始少量丢失，坐标和深度信息不再是100%必备，属性列表衰减加快。
        # ==============================================================================
        # 自然语言描述采样概率
        "description": {"ratio": 1.0},
        # 场景、氛围等属性的简单随机采样率
        "scene_properties": {
            "setting": 1.0,
            "atmosphere": 0.5,
            "lighting": 0.5,
            "style": {"ratio": 0.2, "sample_if_not_contains": ("photorealistic", 1.0)},
        },
        # 摄影属性的采样配置
        "photography": {
            "layout": {"ratio": 0.2, "sample_if_not_contains": ("single image", 1.0)},
            "shot_type": {"ratio": 0.2, "sample_if_not_contains": (None, 1.0)},
            "camera_angle": {"ratio": 0.2, "sample_if_not_contains": ("eye", 1.0)},
            "lens_and_effect": {"ratio": 0.2, "sample_if_not_contains": ("standard lens with deep focus", 1.0)},
        },
        # 元素内部摄影属性的采样率
        "element_photography": {
            "composition": {"ratio": 0.2, "sample_if_not_contains": ("full", 1.0)},
            "view_perspective": {"ratio": 0.5, "sample_if_not_contains": ("front", 1.0)},
            "camera_angle": {"ratio": 0.2, "sample_if_not_contains": ("eye", 1.0)},
            "visibility": {"ratio": 0.2, "sample_if_not_contains": (None, 1.0)},
        },
        # 列表采样的概率模型参数
        "list_sampling": {
            "attributes_actions": {"strength": 1.0, "decay": 0.65, "scale": 1.0}, # 平均约保留74%
            "property_ratio": 0.75, # 75%的概率采样属性列表
            "relationships": {"strength": 1.0, "decay": 0.5, "scale": 1.0},
            "drop_background_element_ratio": 0.5, # 50%的背景元素被丢弃
            "drop_foreground_element_ratio": 0.2, # 20%的前景元素被丢弃
            "drop_position_ratio": 0.2, # 20%的概率不带坐标
            "drop_depth_ratio": 0.2, # 20%的概率不带深度
            "drop_relationship_ratio": 0.5, # 50%的概率丢弃关系
        }
    },
    "l7": {
        # ==============================================================================
        # L7: 标准描述 (Standard Description)
        # 目的：从“机器数据”向“人类自然描述”的转折点。
        # 特征：背景元素丢失一半，属性只保留一半，坐标和深度信息进一步减少。
        #       但注意：前景主体（Foreground）依然全部保留。
        # ==============================================================================
        # 自然语言描述采样概率
        "description": {"ratio": 1.0},
        # 场景、氛围等属性的简单随机采样率
        "scene_properties": {
            "setting": 1.0,
            "atmosphere": 0.5,
            "lighting": 0.5,
            "style": {"ratio": 0.2, "sample_if_not_contains": ("photorealistic", 1.0)},
        },
        # 摄影属性的采样配置
        "photography": {
            "layout": {"ratio": 0.2, "sample_if_not_contains": ("single image", 1.0)},
            "shot_type": {"ratio": 0.2, "sample_if_not_contains": (None, 1.0)},
            "camera_angle": {"ratio": 0.2, "sample_if_not_contains": ("eye", 1.0)},
            "lens_and_effect": {"ratio": 0.2, "sample_if_not_contains": ("standard lens with deep focus", 1.0)},
        },
        # 元素内部摄影属性的采样率
        "element_photography": {
            "composition": {"ratio": 0.2, "sample_if_not_contains": ("full", 1.0)},
            "view_perspective": {"ratio": 0.5, "sample_if_not_contains": ("front", 1.0)},
            "camera_angle": {"ratio": 0.2, "sample_if_not_contains": ("eye", 1.0)},
            "visibility": {"ratio": 0.2, "sample_if_not_contains": (None, 1.0)},
        },
        # 列表采样的概率模型参数
        "list_sampling": {
            "attributes_actions": {"strength": 1.0, "decay": 0.65, "scale": 1.0},
            "property_ratio": 0.5, # 50%的概率采样属性列表
            "relationships": {"strength": 1.0, "decay": 0.5, "scale": 1.0},
            "drop_background_element_ratio": 0.5, # 50%的背景元素被丢弃
            "drop_position_ratio": 0.3, # 30%的概率不带坐标
            "drop_depth_ratio": 0.3, # 30%的概率不带深度
            "drop_relationship_ratio": 0.5,
        }
    },
    "l6": {
        # ==============================================================================
        # L6: 核心构图 (Core Composition)
        # 目的：保留画面的主要内容，去除大部分干扰信息。
        # 特征：背景大幅减少(60%)，属性稀疏。
        #       【关键点】前景主体依然完整保留，确保构图的稳定性。
        # ==============================================================================
        "description": {"ratio": 0.75}, # bbox_json不存在时75%的概率采样描述
        "scene_properties": { # 场景属性更大概率丢失
            "setting": 1.0,
            "atmosphere": 0.3,
            "lighting": 0.3,
            "style": {"ratio": 0.1, "sample_if_not_contains": ("photorealistic", 0.75)},
        },
        "photography": { # 非常规摄影参数有概率丢失
            "layout": {"ratio": 0.1, "sample_if_not_contains": ("single", 0.75)},
            "shot_type": {"ratio": 0.1, "sample_if_not_contains": (None, 0.75)},
            "camera_angle": {"ratio": 0.1, "sample_if_not_contains": ("eye", 0.75)},
            "lens_and_effect": {"ratio": 0.1, "sample_if_not_contains": ("standard", 0.75)},
        },
        "element_photography": { # 非常规摄影参数有概率丢失
            "composition": {"ratio": 0.1, "sample_if_not_contains": ("full", 0.75)},
            "view_perspective": {"ratio": 0.2, "sample_if_not_contains": ("front", 0.75)},
            "camera_angle": {"ratio": 0.1, "sample_if_not_contains": ("eye", 0.75)},
            "visibility": {"ratio": 0.1, "sample_if_not_contains": (None, 0.75)},
        },
        "list_sampling": {
            "attributes_actions": {"strength": 1.0, "decay": 1.0, "scale": 1.0}, # 平均约保留64%
            "property_ratio": 0.5,
            "relationships": {"strength": 1.0, "decay": 1.0, "scale": 1.0},  # 平均约保留64%
            "drop_background_element_ratio": 0.6, # 60%概率丢弃背景元素
            "drop_position_ratio": 0.4, # 40%的概率不带坐标
            "drop_depth_ratio": 0.4, # 40%的概率不带深度
            "drop_relationship_ratio": 0.6, # 60%的概率丢弃关系
        }
    },
    "l5": {
        # ==============================================================================
        # L5: 主体清单 (Subject List)
        # 目的：聚焦于“有什么”和“在哪里”，大幅削减“是什么样子的”。
        # 特征：背景大幅减少(60%)，属性极少，风格/摄影参数不采样。
        #       【关键点】开始引入 10% 的前景丢失，模拟用户描述时的轻微遗漏。
        # ==============================================================================
        "description": {"ratio": 0.75},
        "scene_properties": { # 场景属性更大概率丢失
            "setting": 1.0,
            "atmosphere": 0.3,
            "lighting": 0.3,
            "style": {"ratio": 0.1, "sample_if_not_contains": ("photorealistic", 0.5)},
        },
        "photography": { # 非常规摄影参数更大概率丢失
            "layout": {"ratio": 0.1, "sample_if_not_contains": ("single", 0.5)},
            "shot_type": {"ratio": 0.1, "sample_if_not_contains": (None, 0.5)},
            "camera_angle": {"ratio": 0.1, "sample_if_not_contains": ("eye", 0.5)},
            "lens_and_effect": {"ratio": 0.1, "sample_if_not_contains": ("standard", 0.5)},
        },
        "element_photography": { # 非常规摄影参数更大概率丢失
            "composition": {"ratio": 0.1, "sample_if_not_contains": ("full", 0.5)},
            "view_perspective": {"ratio": 0.2, "sample_if_not_contains": ("front", 0.5)},
            "camera_angle": {"ratio": 0.1, "sample_if_not_contains": ("eye", 0.5)},
            "visibility": {"ratio": 0.1, "sample_if_not_contains": (None, 0.5)},
        },
        "list_sampling": {
            "attributes_actions": {"strength": 0.8, "decay": 1.0, "scale": 1.0}, # 平均约保留51%
            "property_ratio": 0.5,
            "relationships": {"strength": 0.8, "decay": 1.0, "scale": 1.0}, # 平均约保留51%
            "drop_background_element_ratio": 0.6,
            "drop_foreground_element_ratio": 0.1, # 【关键】10%概率丢弃前景元素
            "drop_position_ratio": 0.5, # 50%的概率不带坐标
            "drop_depth_ratio": 0.5, # 50%的概率不带深度
            "drop_relationship_ratio": 0.7, # 70%的概率丢弃关系
        }
    },
    "l4": {
        # ==============================================================================
        # L4: 基础物体 (Basic Objects)
        # 目的：纯粹的物体列表，带有少量位置信息，无属性修饰。
        # 特征：完全不采样属性，不采样关系，坐标信息开始大量丢失。
        #       【关键点】前景丢失率上升至 20%，背景丢失率 70%。
        # ==============================================================================
        "description": {"ratio": 0.5}, # bbox_json不存在时50%的概率采样描述
        "scene_properties": { # 场景属性更大概率丢失
            "setting": 0.8,
            "atmosphere": 0.3,
            "lighting": 0.3,
            "style": {"ratio": 0.1, "sample_if_not_contains": ("photorealistic", 0.5)},
        },
        "photography": {
            "layout": {"ratio": 0.1, "sample_if_not_contains": ("single", 0.5)},
            "shot_type": {"ratio": 0.1, "sample_if_not_contains": (None, 0.5)},
            "camera_angle": {"ratio": 0.1, "sample_if_not_contains": ("eye", 0.5)},
            "lens_and_effect": {"ratio": 0.1, "sample_if_not_contains": ("standard", 0.5)},
        },
        "element_photography": {
            "composition": {"ratio": 0.1, "sample_if_not_contains": ("full", 0.5)},
            "view_perspective": {"ratio": 0.2, "sample_if_not_contains": ("front", 0.5)},
            "camera_angle": {"ratio": 0.1, "sample_if_not_contains": ("eye", 0.5)},
            "visibility": {"ratio": 0.1, "sample_if_not_contains": (None, 0.5)},
        },
        "list_sampling": {
            "attributes_actions": {"strength": 0.8, "decay": 1.0, "scale": 1.0},
            "property_ratio": 0.4, # 40%的概率采样属性列表
            "relationships": {"strength": 0.8, "decay": 1.0, "scale": 1.0},
            "drop_background_element_ratio": 0.7, # 70%概率丢弃背景元素
            "drop_foreground_element_ratio": 0.2, # 【关键】20%概率丢弃前景元素
            "drop_position_ratio": 0.6, # 60%的概率不带坐标
            "drop_depth_ratio": 0.6, # 60%的概率不带深度
            "drop_relationship_ratio": 0.8, # 80%的概率丢弃关系
        }
    },
    "l3": {
        # ==============================================================================
        # L3: 纯名词 (Nouns Only)
        # 目的：只有主体名称（Description/Caption），无位置，无场景。
        # 特征：没有坐标(bbox)，没有深度，没有关系，纯粹的“词袋模型”。
        #       【关键点】前景丢失率 30%，背景丢失率 80%，几乎没有空间信息。
        # ==============================================================================
        "description": {"ratio": 0.3}, # bbox_json不存在时30%的概率采样描述
        "scene_properties": { # 场景属性更大概率丢失
            "setting": 0.5,
            "atmosphere": 0.1,
            "lighting": 0.1,
            "style": {"ratio": 0.1, "sample_if_not_contains": ("photorealistic", 0.25)},
        },
        "photography": { # 非常规摄影参数更大概率丢失
            "layout": {"ratio": 0.1, "sample_if_not_contains": ("single", 0.25)},
            "shot_type": {"ratio": 0.1, "sample_if_not_contains": (None, 0.25)},
            "camera_angle": {"ratio": 0.1, "sample_if_not_contains": ("eye", 0.25)},
            "lens_and_effect": {"ratio": 0.1, "sample_if_not_contains": ("standard", 0.25)},
        },
        "element_photography": { # 非常规摄影参数更大概率丢失
            "composition": {"ratio": 0.1, "sample_if_not_contains": ("full", 0.25)},
            "view_perspective": {"ratio": 0.2, "sample_if_not_contains": ("front", 0.25)},
            "camera_angle": {"ratio": 0.1, "sample_if_not_contains": ("eye", 0.25)},
            "visibility": {"ratio": 0.1, "sample_if_not_contains": (None, 0.25)},
        },
        "list_sampling": {
            "attributes_actions": {"strength": 0.6, "decay": 1.0, "scale": 1.0}, # 平均约保留32%
            "property_ratio": 0.3, # 30%的概率采样属性列表
            "relationships": {"strength": 0.6, "decay": 1.0, "scale": 1.0}, # 平均约保留32%
            "drop_background_element_ratio": 0.8, # 80%概率丢弃背景元素
            "drop_foreground_element_ratio": 0.3, # 【关键】30%概率丢弃前景元素
            "drop_position_ratio": 0.7, # 70%的概率不带坐标
            "drop_depth_ratio": 0.7, # 70%的概率不带坐标
            "drop_relationship_ratio": 0.9,
        }
    },
    "l2": {
        # ==============================================================================
        # L2: 碎片信息 (Fragmented Info)
        # 目的：模拟信息严重缺失或极度概括的情况。
        # 特征：大幅丢弃前景主体（50%），只保留最重要的1-2个物体，甚至可能只剩一个词。
        #       【关键点】背景完全丢弃，前景丢失一半，模拟“残缺数据”。
        # ==============================================================================
        "description": {"ratio": 0.3},
        "scene_properties": { # 场景属性更大概率丢失
            "setting": 0.3,
            "atmosphere": 0.1,
            "lighting": 0.1,
            "style": {"ratio": 0.1, "sample_if_not_contains": ("photorealistic", 0.1)},
        },
        "photography": { # 非常规摄影参数更大概率丢失
            "layout": {"ratio": 0.1, "sample_if_not_contains": ("single", 0.1)},
            "shot_type": {"ratio": 0.1, "sample_if_not_contains": (None, 0.1)},
            "camera_angle": {"ratio": 0.1, "sample_if_not_contains": ("eye", 0.1)},
            "lens_and_effect": {"ratio": 0.1, "sample_if_not_contains": ("standard", 0.1)},
        },
        "element_photography": { # 非常规摄影参数更大概率丢失
            "composition": {"ratio": 0.1, "sample_if_not_contains": ("full", 0.1)},
            "view_perspective": {"ratio": 0.1, "sample_if_not_contains": ("front", 0.1)},
            "camera_angle": {"ratio": 0.1, "sample_if_not_contains": ("eye", 0.1)},
            "visibility": {"ratio": 0.1, "sample_if_not_contains": (None, 0.1)},
        },
        "list_sampling": {
            "attributes_actions": {"strength": 0.2, "decay": 1.0, "scale": 1.0}, # 平均约保留13%
            "property_ratio": 0.2, # 20%的概率采样属性列表
            "relationships": {"strength": 0.2, "decay": 1.0, "scale": 1.0}, # 平均约保留13%
            "drop_background_element_ratio": 1.0, # 无背景元素
            "drop_foreground_element_ratio": 0.5, # 【关键】50%概率丢弃前景元素
            "drop_position_ratio": 0.8, # 80%的概率不带坐标
            "drop_depth_ratio": 0.8, # 80%的概率不带坐标
            "drop_relationship_ratio": 1.0,
        }
    },
    "l1": {
        # ==============================================================================
        # L1: 纯意图 (Intent Only)
        # 目的：仅保留 Intent 字段。
        # 特征：所有元素、场景、风格全部丢弃。Elements 列表为空。
        #       【关键点】所有内容采样率归零，只剩 Intent。
        # ==============================================================================
        "description": {"ratio": 0.0}, # 即使采样元素，也不保留描述（双重保险）
        "scene_properties": {
            "setting": 0.0,
            "atmosphere": 0.0,
            "lighting": 0.0,
            "style": {"ratio": 0.0, "sample_if_not_contains": (None, 0.0)},
        },
        "photography": {
            "layout": {"ratio": 0.0, "sample_if_not_contains": (None, 0.0)},
            "shot_type": {"ratio": 0.0, "sample_if_not_contains": (None, 0.0)},
            "camera_angle": {"ratio": 0.0, "sample_if_not_contains": (None, 0.0)},
            "lens_and_effect": {"ratio": 0.0, "sample_if_not_contains": (None, 0.0)},
        },
        "element_photography": {
            "composition": {"ratio": 0.0, "sample_if_not_contains": (None, 0.0)},
            "view_perspective": {"ratio": 0.0, "sample_if_not_contains": (None, 0.0)},
            "camera_angle": {"ratio": 0.0, "sample_if_not_contains": (None, 0.0)},
            "visibility": {"ratio": 0.0, "sample_if_not_contains": (None, 0.0)},
        },
        "list_sampling": {
            "attributes_actions": {"strength": 0, "decay": 1.0, "scale": 1.0},
            "property_ratio": 0.0,
            "relationships": {"strength": 0, "decay": 1.0, "scale": 1.0},
            "drop_background_element_ratio": 1.0,
            "drop_foreground_element_ratio": 1.0, # 【关键】100% 丢弃前景元素 -> elements 列表为空
            "drop_position_ratio": 1.0,
            "drop_depth_ratio": 1.0,
            "drop_relationship_ratio": 1.0,
        }
    },
}


# ==============================================================================
# 2. 核心函数与辅助函数
# ==============================================================================

def monotonic_probs_np(n, strength, decay, epsilon=1e-3, scale=1.0):
    """
    生成一个长度为n的单调递减的概率数组。
    """
    strength = np.clip(strength, 0.0, 1.0)
    decay = max(decay, 1e-9)
    epsilon = np.clip(epsilon, 1e-9, 0.05)

    if n <= 0:
        return np.empty((0,), dtype=np.float64)
    if n == 1:
        return np.array([1.0], dtype=np.float64) * scale

    xs = np.linspace(0.0, 1.0, n)
    curve = np.exp(-decay * xs)
    probs = epsilon + (1.0 - 2.0 * epsilon) * (strength * curve)
    return np.clip(probs * scale, epsilon, 1.0 - epsilon)


def sample_np(probs):
    """
    根据给定的概率数组生成一个布尔掩码（mock实现）。
    """
    return np.random.rand(len(probs)) < probs


def get_sampled_indices(item_list, config):
    """
    辅助函数：根据单调概率模型获取采样后的索引列表。
    """
    if not item_list:
        return np.array([], dtype=int)
    
    probs = monotonic_probs_np(n=len(item_list), **config)
    mask = sample_np(probs)
    return np.where(mask)[0]


def check_related_elements(sentence, full_id_set):
    """
    检查句子中是否包含相关元素的ID。
    """
    related_id_set = set()
    bracket_contents = re.findall(r'\[(.*?)\]', sentence)
    
    for content in bracket_contents:
        content = content.strip()
        # 识别类似 [1-3] 或 [1–3] 范围
        if range_match := re.match(r'(\d+)\s*[-–]\s*(\d+)', content):
            start, end = map(int, range_match.groups())
            related_id_set.update(range(start, end + 1))
            continue
        # 否则识别逗号/空格分隔的多个数字
        if nums := re.findall(r'\d+', content):
            related_id_set.update(int(n) for n in nums)
            
    return related_id_set.issubset(full_id_set)


def sample_from_element(element):
    """
    对单个元素（element）的内部属性进行采样处理。
    """
    processed_element = {"id": element["id"]}

    if element.get("bbox_json") is None:
        # 1. 采样自然语言描述
        description_config = SAMPLING_CONFIG["description"]
        if random.random() < description_config["ratio"] and element.get("original_description") is not None:
            processed_element["description"] = element.get("original_description", "")
        else:
            processed_element["description"] = element.get("category", "")
    else:
        bbox_json = element["bbox_json"]
        processed_element["caption"] = bbox_json.get("short_caption", "")

        # 1. 采样元素内的 'photography' 属性
        if "photography" in bbox_json and bbox_json["photography"] is not None:
            photography_config = SAMPLING_CONFIG["element_photography"]
            processed_photo = {}
            for key, value in bbox_json["photography"].items():
                if key in photography_config and value:
                    processed_key = key.split("_")[-1]
                    if isinstance(value, list):
                        value = value[0]
                    
                    sample_str = photography_config[key].get("sample_if_not_contains")[0]
                    sample_ratio = photography_config[key].get("sample_if_not_contains")[1]

                    if sample_str and sample_str not in value.lower():
                        if random.random() < sample_ratio:
                            processed_photo[processed_key] = value
                    elif random.random() < photography_config[key]["ratio"]:
                        processed_photo[processed_key] = value

            if processed_photo:
                processed_element["photography"] = processed_photo

        # 2. 采样 'static_attributes' 和 'dynamic_actions'
        prob_config = SAMPLING_CONFIG["list_sampling"]["attributes_actions"]
        for prop_type in ["static_attributes", "dynamic_actions"]:
            if prop_type in bbox_json and bbox_json[prop_type] is not None:
                for k, v_list in bbox_json[prop_type].items():
                    if random.random() > SAMPLING_CONFIG["list_sampling"]["property_ratio"]:
                        continue

                    # 将key中下划线替换为空格
                    # simple_key = k.replace("_", " ")
                    simple_key = k
                    if isinstance(v_list, str):
                        processed_element[simple_key] = v_list
                    elif isinstance(v_list, list) and v_list:
                        kept_indices = get_sampled_indices(v_list, prob_config)
                        if len(kept_indices) > 0:
                            # 移除句末的点并用分号连接
                            tmp_list = [v_list[idx].removesuffix(".") for idx in kept_indices]
                            processed_element[simple_key] = "; ".join(tmp_list)

    # 3. 添加位置和深度信息
    drop_all = random.random() < SAMPLING_CONFIG["list_sampling"].get("drop_all_posistion_depth_ratio", 0)
    if element.get("position") is not None and element.get("position").get("bbox") is not None:
        x1, y1, x2, y2 = element["position"]["bbox"]
        if not drop_all and random.random() > SAMPLING_CONFIG["list_sampling"].get("drop_position_ratio"):
            processed_element["position"] = f"<bbox>{int(x1)} {int(y1)} {int(x2)} {int(y2)}</bbox>"
    if element.get("depth") is not None:
        if not drop_all and random.random() > SAMPLING_CONFIG["list_sampling"].get("drop_depth_ratio"):
            processed_element["depth"] = int(element["depth"])

    return processed_element


def collect_sampled_elements(elements, sample_ids):
    """
    根据配置对元素列表进行采样，并返回采样后的元素列表。
    
    Args:
        elements: 元素列表
        sample_ids: 要采样的元素ID集合
    """
    if not elements:
        return []
    return_elements = []
    for element in elements:
        if element['id'] in sample_ids:
            return_elements.append(sample_from_element(element))
    return return_elements


def maybe_sample_value(value, ratio):
    """
    按概率采样并返回值，未命中时返回None。
    支持字符串、list等类型。
    """
    if not value or random.random() >= ratio:
        return None
    return value[0] if isinstance(value, list) else value


def _sample_optional_fields(field_configs):
    """
    根据配置批量抽样一批独立字段，返回非空结果字典。
    """
    sampled_fields = {}
    for key, (value, ratio) in field_configs.items():
        if sampled_value := maybe_sample_value(value, ratio):
            sampled_fields[key] = sampled_value
    return sampled_fields


def _sample_style_field(style_source, style_config):
    """
    根据配置对 style 进行采样。
    """
    sampled_style = None
    sample_str = style_config.get("sample_if_not_contains")[0]
    sample_ratio = style_config.get("sample_if_not_contains")[1]
    if sample_str and sample_str not in style_source.lower():
        if random.random() < sample_ratio:
            sampled_style = style_source
    elif random.random() < style_config["ratio"]:
        sampled_style = style_source
    return sampled_style


def _sample_photography_block(photography_source, photo_config):
    """
    根据配置对顶级摄影属性进行采样。
    """
    sampled_photography = {}
    for key, config in photo_config.items():
        original_value = check_id_brackets(photography_source.get(key))
        if not original_value:
            continue
        if isinstance(original_value, list):
            original_value = original_value[0]
        
        sample_str = config.get("sample_if_not_contains")[0]
        sample_ratio = config.get("sample_if_not_contains")[1]
        # 满足必采条件时直接保留，否则按比例采样
        if sample_str and sample_str not in original_value.lower():
            if random.random() < sample_ratio:
                sampled_photography[key] = original_value
        elif random.random() < config["ratio"]:
            sampled_photography[key] = original_value
    return sampled_photography


def _sample_scene_block(scene, scene_props_config, sample_bg_ids):
    """
    采样 scene 下的 setting 与 background_elements。
    
    Args:
        scene: scene数据
        scene_props_config: scene属性配置
        sample_bg_ids: 采样后的背景元素ID列表
    """
    scene_data = _sample_optional_fields(
        {"setting": (check_id_brackets(scene.get("setting")), scene_props_config["setting"])}
    )

    if bg_elements := collect_sampled_elements(
        scene.get("background_elements", []), sample_bg_ids
    ):
        scene_data["elements"] = bg_elements
    return scene_data


def _sample_relationships(relationships, rel_config, sample_ids, element_bbox, drop_relationship_ratio):
    """
    根据配置采样 relationships，同时确保引用的元素ID已存在。
    
    Args:
        relationships: 关系列表
        rel_config: 关系采样配置
        sample_ids: 采样后的元素ID集合
        element_bbox: 元素bbox字典
        drop_relationship_ratio: 关系采样概率
    """
    rela_indices = get_sampled_indices(relationships, rel_config)
    if len(rela_indices) == 0:
        return []

    return_relationships = []
    for idx in rela_indices:
        if not check_related_elements(relationships[idx], sample_ids):
            continue
        if random.random() > drop_relationship_ratio:
            return_relationships.append(_replace_element_id(relationships[idx], element_bbox))
    return return_relationships


def _replace_element_id(relationship, element_info):
    """
    将关系中的元素ID替换为元素中心点坐标。
    
    Args:
        relationship: 关系字符串，可能包含类似 [1], [1-3], [1,2,3] 的ID引用
        element_info: 字典，键为元素ID，值为中心点坐标字符串，格式如 "<point>x y</point>" 或 "<bbox>x1 y1 x2 y2</bbox>"
    
    Returns:
        替换后的关系字符串
    """
    if not relationship or not element_info:
        return relationship
    
    # 查找所有方括号内容
    bracket_contents = re.findall(r'\[(.*?)\]', relationship)
    
    # 用于存储替换映射：原始方括号内容 -> 替换后的中心点坐标
    replacement_map = {}
    
    for content in bracket_contents:
        content = content.strip()
        related_ids = []
        
        # 识别类似 [1-3] 或 [1–3] 范围
        if range_match := re.match(r'(\d+)\s*[-–]\s*(\d+)', content):
            start, end = map(int, range_match.groups())
            related_ids = list(range(start, end + 1))
        # 否则识别逗号/空格分隔的多个数字
        else:
            nums = re.findall(r'\d+', content)
            related_ids = [int(n) for n in nums]
        
        # 收集所有有效的元素信息
        element_info_list = []
        for elem_id in related_ids:
            if elem_id in element_info:
                element_info_list.append(element_info[elem_id])
        
        # 如果有有效的元素信息，则进行替换
        if element_info_list:
            # 多个元素信息用空格连接
            replacement_map[f"[{content}]"] = " ".join(element_info_list)
    
    # 执行替换
    result = relationship
    for original, replacement in replacement_map.items():
        result = result.replace(original, replacement)
    
    return result


def has_none_values(data):
    """
    检查值中是否包含None，支持字符串、字典、列表的递归检查
    """
    if isinstance(data, str):
        return data is None
    elif isinstance(data, dict):
        return any(has_none_values(v) for v in data.values())
    elif isinstance(data, list):
        return any(has_none_values(item) for item in data)
    else:
        return False


def _clean_empty_values(data):
    """递归清理 None、空列表[]、空字典{}"""
    if not isinstance(data, (dict, list)):
        return data
    # 处理列表：递归清理元素后过滤无效值
    if isinstance(data, list):
        return [v for v in (map(_clean_empty_values, data)) if v not in (None, [], {}, "")]
    # 处理字典：递归清理值后过滤无效键值对
    return {k: v for k, v in ((k, _clean_empty_values(v)) for k, v in data.items()) if v not in (None, [], {}, "")}


def has_id_brackets(value):
    """
    检查值中是否包含[id]标记，支持字符串、字典、列表的递归检查
    """
    if isinstance(value, str):
        return bool(re.search(r'\[[^\]]*\d[^\]]*\]', value))
    elif isinstance(value, dict):
        return any(has_id_brackets(v) for v in value.values())
    elif isinstance(value, list):
        return any(has_id_brackets(item) for item in value)
    else:
        return False


def check_id_brackets(value):
    """
    检查值中是否包含[id]标记，如果包含则返回None，否则返回原值
    """
    if has_id_brackets(value):
        return None
    return value


def renumber_elements(res):
    """
    重新编号elements和scene.elements的id，从1开始按顺序编号
    """
    # 重新编号elements
    for idx, elem in enumerate(res.get("elements", []), start=1):
        elem["id"] = idx
    
    # 重新编号scene.elements
    if "scene" in res and "elements" in res["scene"]:
        # 计算新的起始id（elements的数量 + 1）
        start_id = len(res.get("elements", [])) + 1
        for idx, elem in enumerate(res["scene"]["elements"], start=start_id):
            elem["id"] = idx
    
    return res


def sample_data_pipeline(data, sampling_config_name="l10"):
    """
    对输入的数据字典进行分层采样，生成一个精简后的样本数据。

    Args:
        data: 包含完整场景信息的原始数据字典。

    Returns:
        一个经过随机采样的新的数据字典。
    """

    global SAMPLING_CONFIG
    SAMPLING_CONFIG = SAMPLING_CONFIG_DICT[sampling_config_name]
    list_configs = SAMPLING_CONFIG["list_sampling"]
    
    # 基础信息
    sample_data = {
        "intent": check_id_brackets(data["intent"]),
        "style": _sample_style_field(data["style"], SAMPLING_CONFIG["scene_properties"]["style"])
    }

    scene = data.get("scene", {})
    scene_props_config = SAMPLING_CONFIG["scene_properties"]
    # 1. 采样独立的顶级属性 (atmosphere, lighting)
    top_level_fields = _sample_optional_fields(
        {
            "atmosphere": (scene.get("atmosphere"), scene_props_config["atmosphere"]),
            "lighting": (check_id_brackets(data.get("lighting")), scene_props_config["lighting"]),
        }
    )
    sample_data.update(top_level_fields)

    # 2. 采样顶级摄影属性
    if photography_data := _sample_photography_block(
        data.get("photography", {}), SAMPLING_CONFIG["photography"]
    ):
        sample_data["photography"] = photography_data

    # 3. 采样主元素 与背景元素 (elements & background_elements)
    fg_all_ids = [element['id'] for element in data.get("elements", [])]
    bg_all_ids = [background_element['id'] for background_element in scene.get("background_elements", [])]
    # 随机采样前景元素
    sample_fg_ids = []
    for fg_id in fg_all_ids:
        if random.random() > list_configs.get("drop_foreground_element_ratio", 0):
            sample_fg_ids.append(fg_id)
    # 随机采样背景元素
    sample_bg_ids = []
    for bg_id in bg_all_ids:
        if random.random() > list_configs.get("drop_background_element_ratio", 0):
            sample_bg_ids.append(bg_id)
    sample_ids = sample_fg_ids + sample_bg_ids

    sampled_elements = collect_sampled_elements(data.get("elements", []), sample_fg_ids)
    if sampled_elements:
        sample_data["elements"] = sampled_elements

    # 4. 收集所有 scene 相关的数据
    if scene_data_to_add := _sample_scene_block(
        scene, scene_props_config, sample_bg_ids
    ):
        sample_data["scene"] = scene_data_to_add

    # 5. 采样关系 (relationships)
    # 构造元素信息，用于计算关系
    element_bbox = {}
    all_elements = data.get("elements", []) + scene.get("background_elements", [])
    for element in all_elements:
        bbox = element.get("position", {}).get("bbox", [])
        if bbox:
            element_bbox[element["id"]] = f"<bbox>{int(bbox[0])} {int(bbox[1])} {int(bbox[2])} {int(bbox[3])}</bbox>"
    
    if rela_data_to_add := _sample_relationships(
        data.get("relationships", []), list_configs["relationships"], sample_ids, element_bbox, list_configs.get("drop_relationship_ratio")
    ):
        sample_data["relationships"] = rela_data_to_add

    # 6. 清理 None 值和空列表
    sample_data = _clean_empty_values(sample_data)

    # 7. 检查是否包含[id]标记，并重新编号elements和scene.elements的id
    assert not has_id_brackets(sample_data), "sample_data contains [id] brackets"
    assert not has_none_values(sample_data), "sample_data contains None values"
    sample_data = renumber_elements(sample_data)

    return sample_data


# ==============================================================================
# QT format variant — flat element attributes, string positions, inline bboxes
# ==============================================================================

# Known structural keys on QT elements (not sampled as attributes)
_QT_STRUCTURAL_KEYS = {"id", "caption", "position", "depth", "photography"}


def _sample_qt_element(element):
    """
    对单个 QT 格式元素进行属性采样。

    QT 元素的属性（appearance, action, design_and_material, state, features 等）
    直接作为扁平 key 存在，而非嵌套在 static_attributes / dynamic_actions 中。
    """
    processed = {"id": element["id"]}

    # 1. caption
    description_config = SAMPLING_CONFIG["description"]
    if element.get("caption") and random.random() < description_config["ratio"]:
        processed["caption"] = element["caption"]

    # 2. 元素级 photography
    if "photography" in element and element["photography"]:
        photo_config = SAMPLING_CONFIG["element_photography"]
        processed_photo = {}
        for key, value in element["photography"].items():
            if key in photo_config and value:
                if isinstance(value, list):
                    value = value[0]
                sample_str, sample_ratio = photo_config[key]["sample_if_not_contains"]
                if sample_str and sample_str not in value.lower():
                    if random.random() < sample_ratio:
                        processed_photo[key] = value
                elif random.random() < photo_config[key]["ratio"]:
                    processed_photo[key] = value
        if processed_photo:
            processed["photography"] = processed_photo

    # 3. 扁平属性（appearance, action, state, features, design_and_material 等）
    prob_config = SAMPLING_CONFIG["list_sampling"]["attributes_actions"]
    for key, value in element.items():
        if key in _QT_STRUCTURAL_KEYS or value is None:
            continue
        if random.random() > SAMPLING_CONFIG["list_sampling"]["property_ratio"]:
            continue
        if isinstance(value, str):
            processed[key] = value
        elif isinstance(value, list) and value:
            kept_indices = get_sampled_indices(value, prob_config)
            if len(kept_indices) > 0:
                tmp_list = [str(value[idx]).removesuffix(".") for idx in kept_indices]
                processed[key] = "; ".join(tmp_list)

    # 4. position（已经是 "<bbox>...</bbox>" 字符串）
    drop_all = random.random() < SAMPLING_CONFIG["list_sampling"].get("drop_all_posistion_depth_ratio", 0)
    if element.get("position") and not drop_all:
        if random.random() > SAMPLING_CONFIG["list_sampling"].get("drop_position_ratio", 0):
            processed["position"] = element["position"]

    # 5. depth
    if element.get("depth") is not None:
        if not drop_all and random.random() > SAMPLING_CONFIG["list_sampling"].get("drop_depth_ratio", 0):
            try:
                processed["depth"] = int(element["depth"])
            except (ValueError, TypeError):
                # depth 可能是范围字符串如 "100-120"，取第一个数字
                import re as _re
                m = _re.search(r'\d+', str(element["depth"]))
                if m:
                    processed["depth"] = int(m.group())

    return processed


def _collect_qt_elements(elements, sample_ids):
    """根据 sample_ids 采样 QT 格式元素列表。"""
    if not elements:
        return []
    return [_sample_qt_element(e) for e in elements if e["id"] in sample_ids]


def _sample_qt_relationships(relationships, rel_config, drop_relationship_ratio):
    """
    采样 QT 格式的 relationships。
    QT relationships 已经内嵌了 <bbox> 字符串，无需做 [id] → bbox 替换。
    """
    rela_indices = get_sampled_indices(relationships, rel_config)
    if len(rela_indices) == 0:
        return []

    result = []
    for idx in rela_indices:
        if random.random() > drop_relationship_ratio:
            result.append(relationships[idx])
    return result


def sample_data_pipeline_qt(data, sampling_config_name="l10"):
    """
    QT 格式数据的分层采样 pipeline。

    与 sample_data_pipeline 的区别：
    - element.position 是字符串 "<bbox>x1 y1 x2 y2</bbox>"（非 dict）
    - element 属性扁平存放（appearance, action 等直接作为 key）
    - atmosphere 在顶层
    - 背景元素在 scene.elements（非 scene.background_elements）
    - relationships 已内嵌 <bbox>，无需 ID 替换
    """
    global SAMPLING_CONFIG
    SAMPLING_CONFIG = SAMPLING_CONFIG_DICT[sampling_config_name]
    list_configs = SAMPLING_CONFIG["list_sampling"]

    scene = data.get("scene", {})
    scene_props_config = SAMPLING_CONFIG["scene_properties"]

    # 基础信息
    sample_data = {
        "intent": check_id_brackets(data["intent"]),
        "style": _sample_style_field(data["style"], scene_props_config["style"]),
    }

    # 1. atmosphere（顶层）、lighting
    top_level_fields = _sample_optional_fields({
        "atmosphere": (data.get("atmosphere"), scene_props_config["atmosphere"]),
        "lighting": (check_id_brackets(data.get("lighting")), scene_props_config["lighting"]),
    })
    sample_data.update(top_level_fields)

    # 2. 顶级摄影属性
    if photography_data := _sample_photography_block(
        data.get("photography", {}), SAMPLING_CONFIG["photography"]
    ):
        sample_data["photography"] = photography_data

    # 3. 采样前景元素与背景元素
    fg_all_ids = [e["id"] for e in data.get("elements", [])]
    bg_all_ids = [e["id"] for e in scene.get("elements", [])]

    sample_fg_ids = [
        fid for fid in fg_all_ids
        if random.random() > list_configs.get("drop_foreground_element_ratio", 0)
    ]
    sample_bg_ids = [
        bid for bid in bg_all_ids
        if random.random() > list_configs.get("drop_background_element_ratio", 0)
    ]

    sampled_fg = _collect_qt_elements(data.get("elements", []), sample_fg_ids)
    if sampled_fg:
        sample_data["elements"] = sampled_fg

    # 4. scene block（setting + 背景元素）
    scene_data = _sample_optional_fields({
        "setting": (check_id_brackets(scene.get("setting")), scene_props_config["setting"]),
    })
    sampled_bg = _collect_qt_elements(scene.get("elements", []), sample_bg_ids)
    if sampled_bg:
        scene_data["elements"] = sampled_bg
    if scene_data:
        sample_data["scene"] = scene_data

    # 5. relationships（已含 <bbox>，无需 ID 替换）
    if rela := _sample_qt_relationships(
        data.get("relationships", []),
        list_configs["relationships"],
        list_configs.get("drop_relationship_ratio", 0),
    ):
        sample_data["relationships"] = rela

    # 6. 清理
    sample_data = _clean_empty_values(sample_data)

    assert not has_none_values(sample_data), "sample_data contains None values"
    sample_data = renumber_elements(sample_data)

    return sample_data
