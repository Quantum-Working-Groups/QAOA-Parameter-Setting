from .utils import MethodJSON
from typing import cast


METHOD_OPTMARKER_PLACEHOLDER = "<optimized_marker>"
METHOD_CONFIG_TO_LABELS: dict[MethodJSON, str] = cast(
    dict[MethodJSON, str],
    {
        #
        # 1. Fourier
        "F.json": f"Fourier{METHOD_OPTMARKER_PLACEHOLDER}",
        "FAer.json": "Fourier (Aer)",
        #
        # 2. Fixed Angle
        "FA_no_opt.json": "Fixed Angle†",
        "FA_noOpt.json": "Fixed Angle†",
        "FA_opt.json": f"Fixed Angle{METHOD_OPTMARKER_PLACEHOLDER}",
        #
        # 3. Fixed Angle with Aer
        "FAAer_opt.json": f"Fixed Angle{METHOD_OPTMARKER_PLACEHOLDER} (Aer)",
        # Virtual method for the zeroth iteration. No angle optimisation
        "FAAer_no_opt.json": "Fixed Angle† (Aer)",
        #
        # 4. Interp
        "I.json": "Interp.",
        "IAer.json": "Interp. (Aer)",
        #
        # 5. Linear Ramp
        "LR_opt.json": f"Linear Ramp{METHOD_OPTMARKER_PLACEHOLDER}",
        "LRAer_opt.json": f"Linear Ramp{METHOD_OPTMARKER_PLACEHOLDER} (Aer)",
        #
        # 6. Transition States
        "RTS.json": "Recursive TS",
        "TS.json": "Recursive TS",
        #
        # 7. Trotterised Quantum Annealing
        "TQA_no_opt.json": "TQA",
        "TQA_noOpt.json": "TQA",
        "TQA_opt.json": f"TQA{METHOD_OPTMARKER_PLACEHOLDER}",
        "TQAAer_opt.json": f"TQA{METHOD_OPTMARKER_PLACEHOLDER} (Aer)",
        # Virtual method for the zeroth iteration of TQAAer. No angle optimisation
        "TQAAer_no_opt.json": "TQA (Aer)",
        # 8. PT_AAAM.json
        "PT_AAAM.json": "Param. Transfer",
    },
)
