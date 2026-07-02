from typing import cast

from .types import MethodConfigJSON, MethodJSON


METHOD_ANGLE_OPT_MARKER_PLACEHOLDER = "*"
METHOD_NO_OPT_AT_ALL_MARKER_PLACEHOLDER = "†"
METHOD_CONFIG_TO_LABELS: dict[MethodJSON, str] = cast(
    dict[MethodJSON, str],
    {
        # _opt.json: ramp parameters and angles are optimised
        # _no_opt.json: parameters and angles are NOT optimised
        # (no flag).json: only the parameters are optimised, not the angles
        #
        # 1. Fourier
        # trainer_config_to_method("F_{SV,MPS,PP}_opt.json") -> "F_opt.json"
        "F_opt.json": f"Fourier{METHOD_ANGLE_OPT_MARKER_PLACEHOLDER}",
        # trainer_config_to_method("F_MPSAer_opt.json") strips "_MPS" -> "FAer_opt.json"
        "FAer_opt.json": f"Fourier{METHOD_ANGLE_OPT_MARKER_PLACEHOLDER}",
        #
        # 2. Fixed Angles
        # trainer_config_to_method("FA_{SV,MPS,PP}_no_opt.json") -> "FA_no_opt.json"
        "FA_no_opt.json": f"Fixed Angles{METHOD_NO_OPT_AT_ALL_MARKER_PLACEHOLDER}",
        # trainer_config_to_method("FA_MPSAer_no_opt.json") strips "_MPS" -> "FAAer_opt.json"
        "FAAer_no_opt.json": f"Fixed Angles{METHOD_NO_OPT_AT_ALL_MARKER_PLACEHOLDER}",
        # trainer_config_to_method("FA_{SV,MPS,PP}_opt.json") -> "FA_opt.json"
        "FA_opt.json": f"Fixed Angles{METHOD_ANGLE_OPT_MARKER_PLACEHOLDER}",
        # trainer_config_to_method("FA_MPSAer_opt.json") strips "_MPS" -> "FAAer_opt.json"
        "FAAer_opt.json": f"Fixed Angles{METHOD_ANGLE_OPT_MARKER_PLACEHOLDER}",
        
        #
        # 3. Interp
        # trainer_config_to_method("I_{SV,MPS,PP}_opt.json") -> "I_opt.json"
        "I_opt.json": f"Interp.{METHOD_ANGLE_OPT_MARKER_PLACEHOLDER}",
        # trainer_config_to_method("I_MPSAer_opt.json") strips "_MPS" -> "IAer_opt.json"
        "IAer_opt.json": f"Interp.{METHOD_ANGLE_OPT_MARKER_PLACEHOLDER}",
        # trainer_config_to_method("I_{SV,MPS,PP}_no_opt.json") -> "I_no_opt.json"
        "I_no_opt.json": f"Interp.{METHOD_NO_OPT_AT_ALL_MARKER_PLACEHOLDER}",
        # trainer_config_to_method("I_MPSAer_no_opt.json") strips "_MPS" -> "IAer_no_opt.json"
        "IAer_no_opt.json": f"Interp.{METHOD_NO_OPT_AT_ALL_MARKER_PLACEHOLDER}",
        # trainer_config_to_method("I_{SV,MPS,PP}.json") -> "I.json"
        "I.json": f"Interp.{METHOD_ANGLE_OPT_MARKER_PLACEHOLDER}",
        # trainer_config_to_method("I_MPSAer.json") -> "IAer.json"
        "IAer.json": f"Interp.{METHOD_ANGLE_OPT_MARKER_PLACEHOLDER}",
        #
        # 4. Linear Ramp
        # trainer_config_to_method("LR_{SV,MPS,PP}_opt.json") -> "LR_opt.json"
        "LR_opt.json": f"Linear Ramp{METHOD_ANGLE_OPT_MARKER_PLACEHOLDER}",
        # trainer_config_to_method("LR_MPSAer_opt.json") strips "_MPS" -> "LRAer_opt.json"
        "LRAer_opt.json": f"Linear Ramp{METHOD_ANGLE_OPT_MARKER_PLACEHOLDER}",
        # trainer_config_to_method("LR_{SV,MPS,PP}_no_opt.json") -> "LR_no_opt.json"
        "LR_no_opt.json": f"Linear Ramp{METHOD_NO_OPT_AT_ALL_MARKER_PLACEHOLDER}",
        # trainer_config_to_method("LR_MPSAer_no_opt.json") strips "_MPS" -> "LRAer_no_opt.json"
        "LRAer_no_opt.json": f"Linear Ramp{METHOD_NO_OPT_AT_ALL_MARKER_PLACEHOLDER}",
        # trainer_config_to_method("LR_{SV,MPS,PP}.json") -> "LR.json"
        "LR.json": f"Linear Ramp",
        # trainer_config_to_method("LR_MPSAer.json") strips "_MPS" -> "LRAer.json"
        "LRAer.json": f"Linear Ramp",
        #
        # 5. Recursive Transition States
        # trainer_config_to_method("RTS_{SV,MPS,PP}_opt.json") -> "RTS_opt.json"
        "RTS_opt.json": f"Recursive TS{METHOD_ANGLE_OPT_MARKER_PLACEHOLDER}",
        # trainer_config_to_method("RTS_MPSAer_opt.json") strips "_MPS" -> "RTSAer_opt.json"
        "RTSAer_opt.json": f"Recursive TS{METHOD_ANGLE_OPT_MARKER_PLACEHOLDER}",
        #
        # 6. Trotterised Quantum Annealing
        # trainer_config_to_method("TQA_{SV,MPS,PP}_no_opt.json") -> "TQA_no_opt.json"
        # (also reached via TRAINER_CONFIG_EQUIVALENT_MAPPINGS TQA_*_no_opt.json -> TQA_*_no_opt.json
        #  and TQA_*_noOpt.json -> TQA_*_no_opt.json)
        "TQA_no_opt.json": f"TQA{METHOD_NO_OPT_AT_ALL_MARKER_PLACEHOLDER}",
        # trainer_config_to_method("TQA_MPSAer_no_opt.json") strips "_MPS" -> "TQAAer_no_opt.json"
        "TQAAer_no_opt.json": f"TQA{METHOD_NO_OPT_AT_ALL_MARKER_PLACEHOLDER}",
        # trainer_config_to_method("TQA_{SV,MPS,PP}.json") -> "TQA.json"
        "TQA.json": "TQA",
        # trainer_config_to_method("TQA_MPSAer.json") strips "_MPS" -> "TQAAer.json"
        "TQAAer.json": "TQA",
        # trainer_config_to_method("TQA_{SV,MPS,PP}_opt.json") -> "TQA_opt.json"
        "TQA_opt.json": f"TQA{METHOD_ANGLE_OPT_MARKER_PLACEHOLDER}",
        # trainer_config_to_method("TQA_MPSAer_opt.json") strips "_MPS" -> "TQAAer_opt.json"
        "TQAAer_opt.json": f"TQA{METHOD_ANGLE_OPT_MARKER_PLACEHOLDER}",

        # 7. Parameter Transfer
        # trainer_config_to_method("PT_PP_AAA.json") strips "_PP" -> "PT_AAA.json"
        "PT_AAA.json": "Param. Transfer",
    },
)
"""Mapping between MethodJSON strings and human-readable labels."""

GRAPH_TYPE_ACRONYMS = {
    "erdos_renyi": "ER",
    "random_regular": "RR",
    "heavy_hex": "HH",
    "line_to_full": "LB",
}
"""Mapping between graph type strings and human-readable acronyms."""

OPT_TO_NO_OPT_MAPPING = {
    # Known mappings between Opt and No-Opt methods.
    "FA_MPS_opt.json": "FA_MPS_no_opt.json",
    "FA_PP_opt.json": "FA_PP_no_opt.json",
    "FA_SV_opt.json": "FA_SV_no_opt.json",
    "TQA_MPS_opt.json": "TQA_MPS_no_opt.json",
    "TQA_PP_opt.json": "TQA_PP_no_opt.json",
    "TQA_SV_opt.json": "TQA_SV_no_opt.json",
    # These no-opt methods don't exist, but we give them names anyway.
    "FA_MPSAer_opt.json": "FA_MPSAer_no_opt.json",
    "TQA_MPSAer_opt.json": "TQA_MPSAer_no_opt.json",
}
"""Known mappings from optimised trainer configs/methods to their unoptimised (no angle optimization) versions.


Unoptimised versions are stored in the zeroth iteration of results for the optimised version.
"""

TRAINER_CONFIG_EQUIVALENT_MAPPINGS: dict[str, MethodConfigJSON] = {
    # Mappings from mislabelled trainer config strings to their correct versions.
    # These handle inconsistent naming conventions where underscores were omitted.
    #
    # angleOpt -> angle_opt
    "LR_MPSAer_angleOpt.json": MethodConfigJSON("LR_MPSAer_opt.json"),
    "LR_MPS_angleOpt.json": MethodConfigJSON("LR_MPS_opt.json"),
    "LR_PP_angleOpt.json": MethodConfigJSON("LR_PP_opt.json"),
    "LR_SV_angleOpt.json": MethodConfigJSON("LR_SV_opt.json"),

    "LR_MPSAer_angle_opt.json": MethodConfigJSON("LR_MPSAer_opt.json"),
    "LR_MPS_angle_opt.json": MethodConfigJSON("LR_MPS_opt.json"),
    "LR_PP_angle_opt.json": MethodConfigJSON("LR_PP_opt.json"),
    "LR_SV_angle_opt.json": MethodConfigJSON("LR_SV_opt.json"),

    "LR_MPSAer_opt.json": MethodConfigJSON("LR_MPSAer.json"),
    "LR_MPS_opt.json": MethodConfigJSON("LR_MPS.json"),
    "LR_PP_opt.json": MethodConfigJSON("LR_PP.json"),
    "LR_SV_opt.json": MethodConfigJSON("LR_SV.json"),
    #
    # noOpt -> no_opt
    # FA
    "FA_MPSAer_noOpt.json": MethodConfigJSON("FA_MPSAer_no_opt.json"),
    "FA_MPS_noOpt.json": MethodConfigJSON("FA_MPS_no_opt.json"),
    "FA_PP_noOpt.json": MethodConfigJSON("FA_PP_no_opt.json"),
    "FA_SV_noOpt.json": MethodConfigJSON("FA_SV_no_opt.json"),
    # TQA: noOpt camelCase -> _no_opt underscore (canonical derived config name)
    "TQA_MPSAer_noOpt.json": MethodConfigJSON("TQA_MPSAer.json"),
    "TQA_MPS_noOpt.json": MethodConfigJSON("TQA_MPS.json"),
    "TQA_PP_noOpt.json": MethodConfigJSON("TQA_PP.json"),
    "TQA_SV_noOpt.json": MethodConfigJSON("TQA_SV.json"),

    "TQA_MPSAer_no_opt.json": MethodConfigJSON("TQA_MPSAer.json"),
    "TQA_MPS_no_opt.json": MethodConfigJSON("TQA_MPS.json"),
    "TQA_PP_no_opt.json": MethodConfigJSON("TQA_PP.json"),
    "TQA_SV_no_opt.json": MethodConfigJSON("TQA_SV.json"),
    #
    # Old bare-flag names (no _opt suffix) -> new _opt names
    # Fourier: F_*.json (no flag) was the only Fourier variant in old data.


    "F_MPS.json": MethodConfigJSON("F_MPS_opt.json"),
    "F_MPSAer.json": MethodConfigJSON("F_MPSAer_opt.json"),
    "F_PP.json": MethodConfigJSON("F_PP_opt.json"),
    "F_SV.json": MethodConfigJSON("F_SV_opt.json"),
    # RTS: old bare-flag RTS names -> _opt
    "RTS_MPS.json": MethodConfigJSON("RTS_MPS_opt.json"),
    "RTS_MPSAer.json": MethodConfigJSON("RTS_MPSAer_opt.json"),
    "RTS_PP.json": MethodConfigJSON("RTS_PP_opt.json"),
    "RTS_SV.json": MethodConfigJSON("RTS_SV_opt.json"),
    # TS: old TS name (pre-rename to RTS) -> new RTS _opt names
    "TS_SV.json": MethodConfigJSON("RTS_SV_opt.json"),
    # PT: old PT_AAAM variant
    "PT_PP_AAAM.json": MethodConfigJSON("PT_PP_AAA.json"),
}
"""Mapping from mislabelled trainer config strings to their correct versions.

This handles inconsistent naming conventions in trainer config filenames where
underscores were omitted (e.g., 'angleOpt' instead of 'angle_opt', 'noOpt'
instead of 'no_opt'). Used to normalize config names when loading results.
"""
