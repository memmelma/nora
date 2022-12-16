from .push_env import (
    PushRedEnv,
    PushGreenEnv,
    PushBlueEnv,
    PushMultitaskEnv,
    PushMultitaskObservableEnv,
)
from .physics_env import (
    PushEnv
)

TABLETOP_ENVIRONMENTS = {
    "push-red": PushRedEnv,
    "push-green": PushGreenEnv,
    "push-blue": PushBlueEnv,
    "push-multitask": PushMultitaskEnv,
    "push-multitask-observable": PushMultitaskObservableEnv,
    "push-dynamic": PushEnv,
}
