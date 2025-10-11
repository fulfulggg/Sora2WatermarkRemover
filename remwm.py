import sys
# HF Hub compat: cached_download shim
try:
    import huggingface_hub as hfh
    if not hasattr(hfh, "cached_download"):
        hfh.cached_download = hfh.hf_hub_download
except Exception:
    pass

import click
from pathlib import Path
from collections import deque
import cv2
import numpy as np
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForCausalLM
from iopaint.model_manager import ModelManager
from iopaint.schema import HDStrategy, LDMSampler, InpaintRequest as Config
import torch
from torch.nn import Module
import tqdm
from loguru import logger
from enum import Enum
import os
import tempfile
import shutil
import subprocess

try:
    from cv2.typing import MatLike
except ImportError:
    MatLike = np.ndarray

class TaskType(str, Enum):
    OPEN_VOCAB_DETECTION = "<OPEN_VOCABULARY_DETECTION>"
    """Detect bounding box for objects and OCR text"""

def identify(task_prompt: TaskType, image: MatLike, text_input: str, model: AutoModelForCausalLM, processor: AutoProcessor, device: str):
    if not isinstance(task_prompt, TaskType):
        raise ValueError(f"task_prompt must be a TaskType, but {task_prompt} is of type {type(task_prompt)}")

    prompt = task_prompt.value if text_input is None else task_prompt.value + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return processor.post_process_generation(
        generated_text, task=task_prompt.value, image_size=(image.width, image.height)
    )

def get_watermark_mask(image: MatLike, model: AutoModelForCausalLM, processor: AutoProcessor, device: str, max_bbox_percent: float, detect_text: str = None, no_white_filter: bool = False, debug_viz_dir: Path = None, debug_frame_idx: int = None, white_s_thresh: int = 80, white_v_thresh: int = 180):
    prompts = []
    if detect_text:
        prompts = [p.strip() for p in str(detect_text).split("|") if p.strip()]
    if not prompts:
        prompts = ["watermark Sora logo", "Sora watermark", "logo", "watermark"]
    task_prompt = TaskType.OPEN_VOCAB_DETECTION

    parsed_answer = {}
    raw_bboxes = []
    kept_bboxes = []
    skipped_size = 0
    skipped_color = 0
    for ptxt in prompts:
        parsed_answer = identify(task_prompt, image, ptxt, model, processor, device)
        detection_key = "<OPEN_VOCABULARY_DETECTION>"
        if detection_key in parsed_answer and "bboxes" in parsed_answer[detection_key]:
            raw_bboxes = parsed_answer[detection_key]["bboxes"] or []
            if raw_bboxes:
                break

    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)

    # Debug overlay (draw accepted boxes)
    overlay = image.copy()
    overlay_draw = ImageDraw.Draw(overlay)

    detection_key = "<OPEN_VOCABULARY_DETECTION>"
    if detection_key in parsed_answer and "bboxes" in parsed_answer[detection_key]:
        image_area = image.width * image.height
        # 画像をnumpy配列に変換（色チェック用）
        image_np = np.array(image)
        
        for bbox in parsed_answer[detection_key]["bboxes"]:
            x1, y1, x2, y2 = map(int, bbox)
            bbox_area = (x2 - x1) * (y2 - y1)
            
            # サイズチェック
            if (bbox_area / image_area) * 100 > max_bbox_percent:
                logger.warning(f"Skipping large bounding box: {bbox} covering {bbox_area / image_area:.2%} of the image")
                skipped_size += 1
                continue
            
            use_box = True
            if not no_white_filter:
                # 色フィルタ：白っぽい領域のみを採用
                roi = image_np[y1:y2, x1:x2]
                if roi.size == 0:
                    logger.warning(f"Empty ROI for bbox: {bbox}")
                    use_box = False
                else:
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
                    # 平均彩度(S)と平均明度(V)を計算
                    avg_saturation = np.mean(hsv_roi[:, :, 1])  # S値（0-255）
                    avg_value = np.mean(hsv_roi[:, :, 2])       # V値（0-255）
                    # 白っぽい判定：彩度が低く（<80）、明度が高い（>180）
                    if avg_saturation < white_s_thresh and avg_value > white_v_thresh:
                        logger.info(f"White watermark detected: S={avg_saturation:.1f}, V={avg_value:.1f}")
                        use_box = True
                    else:
                        logger.info(f"Skipping colored logo by white-filter: S={avg_saturation:.1f}, V={avg_value:.1f}")
                        skipped_color += 1
                        use_box = False

            if use_box:
                draw.rectangle([x1, y1, x2, y2], fill=255)
                kept_bboxes.append((x1, y1, x2, y2))
                try:
                    overlay_draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
                except Exception:
                    pass

    logger.info(f"Detection summary: raw={len(raw_bboxes)} kept={len(kept_bboxes)} skipped_size={skipped_size} skipped_color={skipped_color}")
    # Debug visualization (save roughly every 30 frames to limit IO)
    try:
        if debug_viz_dir is not None and (debug_frame_idx is None or int(debug_frame_idx) % 30 == 0):
            debug_viz_dir.mkdir(parents=True, exist_ok=True)
            idx = 0 if debug_frame_idx is None else int(debug_frame_idx)
            overlay_path = debug_viz_dir / f"frame_{idx:06d}_bbox.jpg"
            mask_path = debug_viz_dir / f"frame_{idx:06d}_mask.png"
            overlay.convert("RGB").save(overlay_path, format="JPEG")
            mask.save(mask_path, format="PNG")
    except Exception as e:
        logger.warning(f"Debug viz save failed: {e}")

    return mask

def process_image_with_lama(image: MatLike, mask: MatLike, model_manager: ModelManager):
    config = Config(
        ldm_steps=50,
        ldm_sampler=LDMSampler.ddim,
        hd_strategy=HDStrategy.CROP,
        hd_strategy_crop_margin=64,
        hd_strategy_crop_trigger_size=800,
        hd_strategy_resize_limit=1600,
    )
    result = model_manager(image, mask, config)

    if result.dtype in [np.float64, np.float32]:
        result = np.clip(result, 0, 255).astype(np.uint8)

    return result

def make_region_transparent(image: Image.Image, mask: Image.Image):
    image = image.convert("RGBA")
    mask = mask.convert("L")
    transparent_image = Image.new("RGBA", image.size)
    for x in range(image.width):
        for y in range(image.height):
            if mask.getpixel((x, y)) > 0:
                transparent_image.putpixel((x, y), (0, 0, 0, 0))
            else:
                transparent_image.putpixel((x, y), image.getpixel((x, y)))
    return transparent_image

def is_video_file(file_path):
    """Check if the file is a video based on its extension"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    return Path(file_path).suffix.lower() in video_extensions

def process_video(input_path, output_path, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format, frame_step=1, target_fps=0.0, temporal_mask=3, mask_dilate=4, no_white_filter=False, detect_text=None, debug_viz_dir=None, white_s_thresh=80, white_v_thresh=180):
    """Process a video file by extracting frames, removing watermarks, and reconstructing the video"""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        logger.error(f"Error opening video file: {input_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_out = target_fps if target_fps > 0 else fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Determine output format
    if force_format:
        output_format = force_format.upper()
    else:
        output_format = "MP4"  # Default to MP4 for videos
    
    # Create output video file
    output_path = Path(output_path)
    if output_path.is_dir():
        output_file = output_path / f"{input_path.stem}_no_watermark.{output_format.lower()}"
    else:
        output_file = output_path.with_suffix(f".{output_format.lower()}")
    
    # Créer un fichier temporaire pour la vidéo sans audio
    temp_dir = tempfile.mkdtemp()
    temp_video_path = Path(temp_dir) / f"temp_no_audio.{output_format.lower()}"
    
    # Set codec based on output format
    if output_format.upper() == "MP4":
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif output_format.upper() == "AVI":
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Default to MP4
    
    out = cv2.VideoWriter(str(temp_video_path), fourcc, fps_out, (width, height))
    
    # Initialize temporal mask buffer and dilation kernel
    temporal_masks = deque(maxlen=max(1, temporal_mask))
    k = max(1, int(mask_dilate))
    kernel = np.ones((k, k), np.uint8) if k > 1 else None

    # Process each frame
    with tqdm.tqdm(total=total_frames, desc="Processing video frames") as pbar:
        frame_count = 0
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Frame skip control
            if frame_idx % frame_step != 0:
                frame_idx += 1
                pbar.update(1)
                progress = int((frame_idx / total_frames) * 100)
                print(f"Processing frame {frame_idx}/{total_frames}, progress:{progress}%")
                continue
            
            # Convert frame to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Get watermark mask (PIL L)
            mask_image = get_watermark_mask(
                pil_image,
                florence_model,
                florence_processor,
                device,
                max_bbox_percent,
                detect_text=detect_text,
                no_white_filter=no_white_filter,
                debug_viz_dir=debug_viz_dir,
                debug_frame_idx=frame_idx,
                white_s_thresh=white_s_thresh,
                white_v_thresh=white_v_thresh
            )

            # Binarize -> temporal fuse -> dilate
            mask_np = np.array(mask_image, dtype=np.uint8)
            mask_bin = (mask_np > 0).astype(np.uint8) * 255
            temporal_masks.append(mask_bin)
            fused_mask = temporal_masks[0].copy()
            for _m in list(temporal_masks)[1:]:
                fused_mask = cv2.bitwise_or(fused_mask, _m)
            if kernel is not None:
                fused_mask = cv2.dilate(fused_mask, kernel, iterations=1)
            
            # Process frame
            if transparent:
                # For video, we can't use transparency, so we'll fill with a color or background
                result_image = make_region_transparent(pil_image, mask_image)
                # Convert RGBA to RGB by filling transparent areas with white
                background = Image.new("RGB", result_image.size, (255, 255, 255))
                background.paste(result_image, mask=result_image.split()[3])
                result_image = background
            else:
                lama_result = process_image_with_lama(np.array(pil_image), fused_mask, model_manager)
                result_image = Image.fromarray(cv2.cvtColor(lama_result, cv2.COLOR_BGR2RGB))
            
            # Convert back to OpenCV format and write to output video
            frame_result = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
            out.write(frame_result)
            
            # Update progress
            frame_count += 1
            frame_idx += 1
            pbar.update(1)
            progress = int((frame_idx / total_frames) * 100)
            print(f"Processing frame {frame_idx}/{total_frames}, progress:{progress}%")
    
    # Release resources
    cap.release()
    out.release()
    
    # Combiner la vidéo traitée avec l'audio original à l'aide de FFmpeg
    try:
        logger.info("Fusion de la vidéo traitée avec l'audio original...")
        
        # Vérifier si FFmpeg est disponible
        try:
            subprocess.check_output(["ffmpeg", "-version"], stderr=subprocess.STDOUT)
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("FFmpeg n'est pas disponible. La vidéo sera produite sans audio.")
            shutil.copy(str(temp_video_path), str(output_file))
        else:
            # Utiliser FFmpeg pour combiner la vidéo traitée avec l'audio original
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", str(temp_video_path),  # Vidéo traitée sans audio
                "-i", str(input_path),       # Vidéo originale avec audio
                "-c:v", "copy",              # Copier la vidéo sans réencodage
                "-c:a", "aac",               # Encoder l'audio en AAC pour meilleure compatibilité
                "-map", "0:v:0",             # Utiliser la piste vidéo du premier fichier (vidéo traitée)
                "-map", "1:a:0",             # Utiliser la piste audio du deuxième fichier (vidéo originale)
                "-shortest",                  # Terminer quand la piste la plus courte se termine
                str(output_file)
            ]
            
            # Exécuter FFmpeg
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info("Fusion audio/vidéo terminée avec succès!")
    except Exception as e:
        logger.error(f"Erreur lors de la fusion audio/vidéo: {str(e)}")
        # En cas d'erreur, utiliser la vidéo sans audio
        shutil.copy(str(temp_video_path), str(output_file))
    finally:
        # Nettoyer les fichiers temporaires
        try:
            os.remove(str(temp_video_path))
            os.rmdir(temp_dir)
        except:
            pass
    
    logger.info(f"input_path:{input_path}, output_path:{output_file}, overall_progress:100")
    return output_file

def handle_one(image_path: Path, output_path: Path, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format, overwrite, frame_step=1, target_fps=0.0, temporal_mask=3, mask_dilate=4, no_white_filter: bool = False, detect_text: str = None, debug_viz_dir: Path = None, white_s_thresh: int = 80, white_v_thresh: int = 180):
    if output_path.exists() and not overwrite:
        logger.info(f"Skipping existing file: {output_path}")
        return

    # Check if it's a video file
    if is_video_file(image_path):
        return process_video(image_path, output_path, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format, frame_step, target_fps, temporal_mask, mask_dilate, no_white_filter, detect_text, debug_viz_dir, white_s_thresh, white_v_thresh)

    # Process image
    image = Image.open(image_path).convert("RGB")
    mask_image = get_watermark_mask(image, florence_model, florence_processor, device, max_bbox_percent, detect_text=detect_text, no_white_filter=no_white_filter, debug_viz_dir=debug_viz_dir, debug_frame_idx=0, white_s_thresh=white_s_thresh, white_v_thresh=white_v_thresh)

    if transparent:
        result_image = make_region_transparent(image, mask_image)
    else:
        lama_result = process_image_with_lama(np.array(image), np.array(mask_image), model_manager)
        result_image = Image.fromarray(cv2.cvtColor(lama_result, cv2.COLOR_BGR2RGB))

    # Determine output format
    if force_format:
        output_format = force_format.upper()
    elif transparent:
        output_format = "PNG"
    else:
        output_format = image_path.suffix[1:].upper()
        if output_format not in ["PNG", "WEBP", "JPG"]:
            output_format = "PNG"
    
    # Map JPG to JPEG for PIL compatibility
    if output_format == "JPG":
        output_format = "JPEG"

    if transparent and output_format == "JPG":
        logger.warning("Transparency detected. Defaulting to PNG for transparency support.")
        output_format = "PNG"

    new_output_path = output_path.with_suffix(f".{output_format.lower()}")
    result_image.save(new_output_path, format=output_format)
    logger.info(f"input_path:{image_path}, output_path:{new_output_path}")
    return new_output_path

@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--overwrite", is_flag=True, help="Overwrite existing files in bulk mode.")
@click.option("--transparent", is_flag=True, help="Make watermark regions transparent instead of removing.")
@click.option("--max-bbox-percent", default=10.0, help="Maximum percentage of the image that a bounding box can cover.")
@click.option("--force-format", type=click.Choice(["PNG", "WEBP", "JPG", "MP4", "AVI"], case_sensitive=False), default=None, help="Force output format. Defaults to input format.")
@click.option("--frame-step", default=1, type=int, help="Process every Nth frame (1=all frames, 2=every other frame)")
@click.option("--target-fps", default=0.0, type=float, help="Target output FPS (0=same as input)")
@click.option("--temporal-mask", "opt_temporal_mask", default=3, type=int, help="Temporal mask window size (frames).")
@click.option("--mask-dilate", "opt_mask_dilate", default=4, type=int, help="Mask dilation size (px).")
@click.option("--no-white-filter", is_flag=True, help="Disable white logo color filter (S<V and V>threshold).")
@click.option("--white-s-thresh", default=80, type=int, help="White-filter saturation threshold S_max (default: 80).")
@click.option("--white-v-thresh", default=180, type=int, help="White-filter brightness/value threshold V_min (default: 180).")
@click.option("--detect-text", default=None, type=str, help='Override detection prompt for Florence-2 (default: "watermark Sora logo").')
@click.option("--debug-viz-dir", type=click.Path(file_okay=False), default=None, help="Directory to save debug visualization images (every ~30 frames).")
def main(input_path: str, output_path: str, overwrite: bool, transparent: bool, max_bbox_percent: float, force_format: str, frame_step: int, target_fps: float, opt_temporal_mask: int, opt_mask_dilate: int, no_white_filter: bool, detect_text: str, debug_viz_dir: str, white_s_thresh: int, white_v_thresh: int):
    # Input validation
    if frame_step < 1:
        logger.error("frame_step must be >= 1")
        sys.exit(1)
    if target_fps < 0:
        logger.error("target_fps must be >= 0")
        sys.exit(1)
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    dbg_dir = Path(debug_viz_dir) if debug_viz_dir else None
    if dbg_dir:
        dbg_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    florence_model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True).to(device).eval()
    florence_processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
    logger.info("Florence-2 Model loaded")

    if not transparent:
        try:
            model_manager = ModelManager(name="lama", device=device)
            logger.info("LaMa model loaded")
        except NotImplementedError as e:
            logger.warning(
                f"LaMa backend is not available in this environment: {e}. "
                "Falling back to OpenCV inpaint backend (cv2). "
                "To enable LaMa, reinstall iopaint with LaMa support and restart the runtime."
            )
            model_manager = ModelManager(name="cv2", device=device)
            logger.info("cv2 inpaint backend loaded")
    else:
        model_manager = None

    if input_path.is_dir():
        if not output_path.exists():
            output_path.mkdir(parents=True)

        # Include video files in the search
        images = list(input_path.glob("*.[jp][pn]g")) + list(input_path.glob("*.webp"))
        videos = list(input_path.glob("*.mp4")) + list(input_path.glob("*.avi")) + list(input_path.glob("*.mov")) + list(input_path.glob("*.mkv"))
        files = images + videos
        total_files = len(files)

        for idx, file_path in enumerate(tqdm.tqdm(files, desc="Processing files")):
            output_file = output_path / file_path.name
            handle_one(file_path, output_file, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format, overwrite, frame_step, target_fps, opt_temporal_mask, opt_mask_dilate, no_white_filter, detect_text, dbg_dir, white_s_thresh, white_v_thresh)
            progress = int((idx + 1) / total_files * 100)
            print(f"input_path:{file_path}, output_path:{output_file}, overall_progress:{progress}")
    else:
        output_file = output_path
        if is_video_file(input_path) and output_path.suffix.lower() not in ['.mp4', '.avi', '.mov', '.mkv']:
            # Ensure video output has proper extension
            if force_format and force_format.upper() in ["MP4", "AVI"]:
                output_file = output_path.with_suffix(f".{force_format.lower()}")
            else:
                output_file = output_path.with_suffix(".mp4")  # Default to mp4
        
        handle_one(input_path, output_file, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format, overwrite, frame_step, target_fps, opt_temporal_mask, opt_mask_dilate, no_white_filter, detect_text, dbg_dir, white_s_thresh, white_v_thresh)
        print(f"input_path:{input_path}, output_path:{output_file}, overall_progress:100")

if __name__ == "__main__":
    main()
