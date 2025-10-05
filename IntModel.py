from typing import List
from enum import Enum
from pydantic import BaseModel, Field

class AspectCategoryEnum(str, Enum):
    team_dynamics = "Team Dynamics"
    workload = "Workload & Staffing"
    equipment = "Equipment & Processes"
    leadership = "Leadership & Management"

class Aspect(BaseModel):
    aspect_term: str = Field(description = "The exact word or phrase in the text that represents a specific feature, attribute, or aspect of a product or service that a user may express an opinion about, the aspect term might be 'NULL' for implicit aspect.")
    aspect_category:  AspectCategoryEnum = Field(description = "Refers to the category that aspect belongs to, and the available categories")
    polarity: str = Field(description = "The degree of positivity, negativity or neutrality expressed in the opinion towards a particular aspect or feature of a product or service, and the available polarities include: 'positive', 'negative' and 'neutral'.")
    opinion_term: str = Field(description = "The exact word or phrase in the text that refers to the sentiment or attitude expressed by a user towards a particular aspect or feature of a product or service, the aspect term might be 'NULL' for implicit opinion.")


class ASQPRecord(BaseModel):
    pred_label: List[Aspect]