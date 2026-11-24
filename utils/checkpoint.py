import random
from pathlib import Path
import numpy as np
import torch


def save_checkpoint(path_name, student, alignment, optimizer, scheduler, state, run_config):
    checkpoint = {"model_state_dict": student.state_dict(), "alignment_state_dict": alignment.state_dict() if alignment is not None else None, "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "state": state, "config": run_config, "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [], }, }
    path_name = Path(path_name)
    temporary = path_name.with_suffix(".writing")
    torch.save(checkpoint, temporary)
    temporary.replace(path_name)


def restore_checkpoint(path_name, student, alignment, optimizer, scheduler):
    checkpoint = torch.load(path_name, map_location="cpu", weights_only=False)
    student.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if alignment is not None:
        alignment.load_state_dict(checkpoint["alignment_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    rng = checkpoint["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    return checkpoint["state"]


def load_student(path, student_name=None):
    from models.model_zoo import build_student

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    name = config.get("student", config.get("Student", student_name))
    if name is None:
        raise ValueError("A raw state dictionary requires --student to identify the architecture.")
    if student_name is not None and name != student_name:
        raise ValueError("The selected student differs from the checkpoint architecture.")
    model = build_student(name)
    state = checkpoint.get("model_state_dict", checkpoint)
    if state and all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model
