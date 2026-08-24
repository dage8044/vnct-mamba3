"""Quality-prediction heads."""

from vnct.models.heads.common import QualityHeadOutput
from vnct.models.heads.joint_stage import JointStageHeadOutput, JointStageQualityHead
from vnct.models.heads.maniqa_weighted import MANIQAPatchWeightedHead

__all__ = [
    "JointStageHeadOutput",
    "JointStageQualityHead",
    "MANIQAPatchWeightedHead",
    "QualityHeadOutput",
]
