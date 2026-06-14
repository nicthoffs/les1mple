from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any


ACTION_BUTTONS = (
    "attack",
    "attack2",
    "jump",
    "duck",
    "forward",
    "back",
    "moveleft",
    "moveright",
    "use",
    "reload",
    "walk",
    "score",
)

TIME_COLUMNS = ("t", "time", "timestamp", "video_time", "video_timestamp")
YAW_COLUMNS = ("yaw", "view_yaw", "viewangle_yaw", "view_angle_yaw")
PITCH_COLUMNS = ("pitch", "view_pitch", "viewangle_pitch", "view_angle_pitch")
MOUSE_X_COLUMNS = ("delta_yaw", "mouse_dx", "mousedx", "mouse_x", "dx")
MOUSE_Y_COLUMNS = ("delta_pitch", "mouse_dy", "mousedy", "mouse_y", "dy")
BUTTON_COLUMNS = ("active", "buttons", "keys", "input_buttons")


def action_dim() -> int:
    """Return the current OpenCS2 action vector width."""

    # yaw_delta, pitch_delta, then button multi-hot.
    return 2 + len(ACTION_BUTTONS)


class SyntheticOpenCS2Dataset:
    """Small fake OpenCS2-shaped dataset for testing the LeWM data contract."""

    def __init__(
        self,
        *,
        num_samples: int = 8,
        sequence_length: int = 8,
        image_size: int = 224,
    ) -> None:
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if image_size < 1:
            raise ValueError("image_size must be positive")

        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.image_size = image_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, object]:
        torch = _require_torch()

        if index < 0 or index >= self.num_samples:
            raise IndexError(index)

        generator = torch.Generator().manual_seed(index)
        pixels = torch.rand(
            self.sequence_length,
            3,
            self.image_size,
            self.image_size,
            generator=generator,
        )

        action = torch.zeros(self.sequence_length, action_dim())
        action[:, 0] = torch.randn(self.sequence_length, generator=generator) * 0.02
        action[:, 1] = torch.randn(self.sequence_length, generator=generator) * 0.02
        action[:, 2 + ACTION_BUTTONS.index("forward")] = 1.0
        action[::4, 2 + ACTION_BUTTONS.index("attack")] = 1.0

        return {
            "pixels": pixels,
            "action": action,
            "sample_id": f"synthetic-{index}",
        }


class LocalTensorSequenceDataset:
    """Dataset for cached OpenCS2 sequence tensors.

    Each ``.pt`` file must contain a dict with:
    - ``pixels``: uint8 ``[0, 255]`` or float ``[0, 1]`` tensor shaped ``T,C,H,W``
    - ``action``: float tensor shaped ``T,A``
    """

    def __init__(
        self,
        data_dir: str | Path,
        *,
        pixel_dtype: str = "float32",
        mmap_load: bool = False,
    ) -> None:
        if pixel_dtype not in {"float32", "uint8"}:
            raise ValueError(f"unknown pixel dtype {pixel_dtype!r}")
        self.data_dir = Path(data_dir)
        self.pixel_dtype = pixel_dtype
        self.mmap_load = mmap_load
        self.files = sorted(self.data_dir.glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"no .pt sequence files found in {self.data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, object]:
        torch = _require_torch()

        path = self.files[index]
        sample = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=self.mmap_load,
        )
        _validate_sample(sample, sample_id=str(path))
        sample["pixels"] = _normalize_pixels_for_model(
            sample["pixels"],
            pixel_dtype=self.pixel_dtype,
        )
        sample.setdefault("sample_id", path.stem)
        return sample

    def action_at(self, index: int):
        torch = _require_torch()

        path = self.files[index]
        sample = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=self.mmap_load,
        )
        action = sample.get("action") if isinstance(sample, dict) else None
        if not isinstance(action, torch.Tensor):
            raise TypeError(f"{path}: action must be a tensor")
        if action.ndim != 2:
            raise ValueError(f"{path}: action must have shape T,A")
        return action


def build_local_sequence(
    *,
    video_path: str | Path,
    ticks_path: str | Path,
    output_path: str | Path,
    start_frame: int = 0,
    sequence_length: int = 8,
    frame_step: int = 1,
    image_size: int = 224,
    time_column: str | None = None,
    buttons_column: str | None = None,
) -> dict[str, object]:
    """Build one LeWM-shaped tensor sequence from local OpenCS2 files."""

    cv2 = _require_cv2()
    pd = _require_pandas()
    torch = _require_torch()

    video_path = Path(video_path)
    ticks_path = Path(ticks_path)
    output_path = Path(output_path)

    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    if frame_step < 1:
        raise ValueError("frame_step must be positive")

    frame_indices = [start_frame + i * frame_step for i in range(sequence_length)]
    pixels, video_fps = _read_video_frames(
        cv2=cv2,
        torch=torch,
        video_path=video_path,
        frame_indices=frame_indices,
        image_size=image_size,
    )

    ticks = pd.read_parquet(ticks_path)
    frame_times = [index / video_fps for index in frame_indices]
    tick_rows = _nearest_tick_rows(
        ticks=ticks,
        frame_times=frame_times,
        time_column=time_column,
    )
    action = _encode_actions(
        torch=torch,
        tick_rows=tick_rows,
        buttons_column=buttons_column,
    )

    sample = {
        "pixels": pixels,
        "action": action,
        "sample_id": output_path.stem,
        "frame_indices": torch.tensor(frame_indices, dtype=torch.int64),
        "frame_times": torch.tensor(frame_times, dtype=torch.float32),
    }
    _validate_sample(sample, sample_id=str(video_path))

    _atomic_torch_save(torch, sample, output_path)
    return sample


def _atomic_torch_save(torch: Any, sample: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = tempfile.NamedTemporaryFile(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        delete=False,
    )
    temp_path = Path(temp_file.name)
    temp_file.close()
    try:
        torch.save(sample, temp_path)
        temp_path.replace(output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _read_video_frames(
    *,
    cv2: Any,
    torch: Any,
    video_path: Path,
    frame_indices: list[int],
    image_size: int,
) -> tuple[object, float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video {video_path}")

    try:
        video_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if video_fps <= 0:
            raise ValueError(f"could not read FPS for {video_path}")

        frames = []
        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(
                    f"could not read frame {frame_index} from {video_path}"
                )
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(
                frame, (image_size, image_size), interpolation=cv2.INTER_AREA
            )
            tensor = torch.from_numpy(frame).permute(2, 0, 1).contiguous()
            frames.append(tensor)

        return torch.stack(frames), video_fps
    finally:
        capture.release()


def _nearest_tick_rows(
    *,
    ticks: Any,
    frame_times: list[float],
    time_column: str | None,
) -> Any:
    time_column = time_column or _first_existing_column(ticks, TIME_COLUMNS)
    if time_column is None:
        raise ValueError(
            f"ticks parquet needs one timestamp column from {', '.join(TIME_COLUMNS)}"
        )

    ordered = ticks.sort_values(time_column).reset_index(drop=True)
    nearest_indices = ordered[time_column].searchsorted(frame_times)
    nearest_indices = nearest_indices.clip(0, len(ordered) - 1)

    for i, frame_time in enumerate(frame_times):
        right = nearest_indices[i]
        left = max(0, right - 1)
        if abs(ordered.loc[left, time_column] - frame_time) <= abs(
            ordered.loc[right, time_column] - frame_time
        ):
            nearest_indices[i] = left

    return ordered.iloc[nearest_indices].reset_index(drop=True)


def _encode_actions(
    *,
    torch: Any,
    tick_rows: Any,
    buttons_column: str | None,
) -> object:
    action = torch.zeros(len(tick_rows), action_dim(), dtype=torch.float32)

    mouse_x = _first_existing_column(tick_rows, MOUSE_X_COLUMNS)
    mouse_y = _first_existing_column(tick_rows, MOUSE_Y_COLUMNS)
    if mouse_x and mouse_y:
        action[:, 0] = torch.tensor(tick_rows[mouse_x].to_numpy(), dtype=torch.float32)
        action[:, 1] = torch.tensor(tick_rows[mouse_y].to_numpy(), dtype=torch.float32)
    else:
        yaw = _first_existing_column(tick_rows, YAW_COLUMNS)
        pitch = _first_existing_column(tick_rows, PITCH_COLUMNS)
        if yaw and pitch:
            yaw_values = torch.tensor(tick_rows[yaw].to_numpy(), dtype=torch.float32)
            pitch_values = torch.tensor(
                tick_rows[pitch].to_numpy(), dtype=torch.float32
            )
            action[1:, 0] = yaw_values[1:] - yaw_values[:-1]
            action[1:, 1] = pitch_values[1:] - pitch_values[:-1]

    buttons_column = buttons_column or _first_existing_column(tick_rows, BUTTON_COLUMNS)
    if buttons_column:
        for row_index, raw_buttons in enumerate(tick_rows[buttons_column]):
            buttons = _normalize_buttons(raw_buttons)
            for button_index, button in enumerate(ACTION_BUTTONS):
                if button in buttons:
                    action[row_index, 2 + button_index] = 1.0

    return action


def _normalize_buttons(value: object) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        pieces = value.replace(",", " ").replace("|", " ").replace(";", " ").split()
    elif isinstance(value, (list, tuple, set)) or hasattr(value, "__iter__"):
        pieces = [str(piece) for piece in value]
    else:
        return set()

    return {piece.lower().lstrip("+").replace("_", "") for piece in pieces}


def _first_existing_column(dataframe: Any, names: tuple[str, ...]) -> str | None:
    columns = set(dataframe.columns)
    for name in names:
        if name in columns:
            return name
    return None


def _validate_sample(sample: object, *, sample_id: str) -> None:
    torch = _require_torch()

    if not isinstance(sample, dict):
        raise TypeError(f"{sample_id}: expected dict sample")

    pixels = sample.get("pixels")
    action = sample.get("action")
    if not isinstance(pixels, torch.Tensor):
        raise TypeError(f"{sample_id}: pixels must be a tensor")
    if not isinstance(action, torch.Tensor):
        raise TypeError(f"{sample_id}: action must be a tensor")
    if pixels.ndim != 4:
        raise ValueError(f"{sample_id}: pixels must have shape T,C,H,W")
    if action.ndim != 2:
        raise ValueError(f"{sample_id}: action must have shape T,A")
    if pixels.shape[0] != action.shape[0]:
        raise ValueError(f"{sample_id}: pixels and action must share sequence length")
    if pixels.shape[1] != 3:
        raise ValueError(f"{sample_id}: pixels channel dimension must be 3")
    if pixels.dtype == torch.uint8:
        return
    if not torch.is_floating_point(pixels):
        raise TypeError(f"{sample_id}: pixels must be uint8 or floating point")


def _normalize_pixels_for_model(pixels, *, pixel_dtype: str = "float32"):
    torch = _require_torch()

    if pixel_dtype == "uint8" and pixels.dtype == torch.uint8:
        return pixels
    if pixels.dtype == torch.uint8:
        return pixels.float() / 255.0
    return pixels.float()


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for LeS1mple datasets. Install project dependencies first."
        ) from exc

    return torch


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required to decode local OpenCS2 videos. Install project dependencies first."
        ) from exc

    return cv2


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "pandas is required to read OpenCS2 tick parquet files. Install project dependencies first."
        ) from exc

    return pd
