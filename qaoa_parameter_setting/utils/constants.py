from typing import cast

from qaoa_parameter_setting.utils.types import MethodJSON


METHOD_ANGLE_OPT_MARKER_PLACEHOLDER = "*"
METHOD_NO_OPT_AT_ALL_MARKER_PLACEHOLDER = "†"
METHOD_CONFIG_TO_LABELS: dict[MethodJSON, str] = cast(
    dict[MethodJSON, str],
    {
        #
        # 1. Fourier
        "F.json": f"Fourier{METHOD_ANGLE_OPT_MARKER_PLACEHOLDER}",
        "FAer.json": f"Fourier{METHOD_ANGLE_OPT_MARKER_PLACEHOLDER}",
        #
        # 2. Fixed Angle
        "FA_no_opt.json": f"Fixed Angle{METHOD_NO_OPT_AT_ALL_MARKER_PLACEHOLDER}",
        "FA_noOpt.json": f"Fixed Angle{METHOD_NO_OPT_AT_ALL_MARKER_PLACEHOLDER}",
        "FA_opt.json": f"Fixed Angle{METHOD_ANGLE_OPT_MARKER_PLACEHOLDER}",
        #
        # 3. Fixed Angle with Aer
        "FAAer_opt.json": f"Fixed Angle{METHOD_ANGLE_OPT_MARKER_PLACEHOLDER}",
        # Virtual method for the zeroth iteration. No angle optimisation
        "FAAer_no_opt.json": f"Fixed Angle{METHOD_NO_OPT_AT_ALL_MARKER_PLACEHOLDER}",
        #
        # 4. Interp
        # The parameters are optimised in the default files.
        "I.json": f"Interp.{METHOD_ANGLE_OPT_MARKER_PLACEHOLDER}",
        "IAer.json": f"Interp.{METHOD_ANGLE_OPT_MARKER_PLACEHOLDER}",
        # Virtual method for the zeroth iteration. No angle optimisation.
        "I_no_opt.json": "Interp.",
        "IAer_no_opt.json": "Interp.",
        #
        # 5. Linear Ramp
        # _opt.json: The angles are not optimised, only the two linear-ramp parameters.
        "LR_opt.json": "Linear Ramp",
        "LRAer_opt.json": "Linear Ramp",
        # _angle_opt.json: The angles ARE optimised.
        "LR_angle_opt.json": f"Linear Ramp{METHOD_ANGLE_OPT_MARKER_PLACEHOLDER}",
        "LRAer_angle_opt.json": f"Linear Ramp{METHOD_ANGLE_OPT_MARKER_PLACEHOLDER}",
        #
        # 6. Transition States
        "RTS.json": "Recursive TS",
        "TS.json": "Recursive TS",
        #
        # 7. Trotterised Quantum Annealing
        "TQA_no_opt.json": "TQA",
        "TQA_noOpt.json": "TQA",
        "TQA_opt.json": f"TQA{METHOD_ANGLE_OPT_MARKER_PLACEHOLDER}",
        "TQAAer_opt.json": f"TQA{METHOD_ANGLE_OPT_MARKER_PLACEHOLDER}",
        # Virtual method for the zeroth iteration of TQAAer. No angle optimisation
        "TQAAer_no_opt.json": "TQA",
        # 8. Parameter Transfer
        "PT_AAAM.json": "Param. Transfer",
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
    "FA_SV_opt.json": "FA_SV_noOpt.json",
    "TQA_MPS_opt.json": "TQA_MPS_no_opt.json",
    "TQA_PP_opt.json": "TQA_PP_no_opt.json",
    "TQA_SV_opt.json": "TQA_SV_noOpt.json",
    # These no-opt methods don't exist, but we give them names anyway.
    "FA_MPSAer_opt.json": "FA_MPSAer_no_opt.json",
    "TQA_MPSAer_opt.json": "TQA_MPSAer_no_opt.json",
}
"""Known mappings from optimised trainer configs/methods to their unoptimised versions.


Unoptimised versions are stored in the zeroth iteration of results for the optimised version.
"""
