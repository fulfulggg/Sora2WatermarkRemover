Do not propose changes to these files, treat them as *read-only*.
If you need to edit any of these files, ask me to *add them to the chat* first.

.github/ISSUE_TEMPLATE/bug_report.md

CODE_OF_CONDUCT.md

DEMARRAGE_RAPIDE.md

INSTALLATION_FR.md

LICENSE

README.md

Sora2WatermarkRemover_Colab.ipynb

create_shortcut.ps1

download_lama.py

environment.yml

install_windows.bat

install_windows.ps1

remwm.py:
⋮
│class TaskType(str, Enum):
⋮
│def identify(task_prompt: TaskType, image: MatLike, text_input: str, model: AutoModelForCausalLM, p
⋮
│def get_watermark_mask(image: MatLike, model: AutoModelForCausalLM, processor: AutoProcessor, devic
⋮
│def process_image_with_lama(image: MatLike, mask: MatLike, model_manager: ModelManager):
⋮
│def make_region_transparent(image: Image.Image, mask: Image.Image):
⋮
│def is_video_file(file_path):
⋮
│def process_video(input_path, output_path, florence_model, florence_processor, model_manager, devic
⋮
│def handle_one(image_path: Path, output_path: Path, florence_model, florence_processor, model_manag
⋮
│@click.command()
⋮
│def main(input_path: str, output_path: str, overwrite: bool, transparent: bool, max_bbox_percent: f
⋮

remwmgui.py:
⋮
│CONFIG_FILE = "ui.yml"
│
│class Worker(QObject):
│    log_signal = pyqtSignal(str)
⋮
│    def __init__(self, process):
⋮
│    def run(self):
⋮
│    def read_stderr(self):
⋮
│    def stop(self):
⋮
│class WatermarkRemoverGUI(QMainWindow):
│    def __init__(self):
│        super().__init__()
│        self.setWindowTitle("Watermark Remover GUI")
│        self.setGeometry(100, 100, 800, 600)
│
│        # Initialize UI elements
│        self.radio_single = QRadioButton("Process Single File")
│        self.radio_batch = QRadioButton("Process Directory")
│        self.radio_single.setChecked(True)
│        self.mode_group = QButtonGroup()
⋮
│    def update_bbox_label(self, value):
⋮
│    def toggle_logs(self, checked):
⋮
│    def apply_dark_mode_if_needed(self):
⋮
│    def update_system_info(self):
⋮
│    def browse_input(self):
⋮
│    def browse_output(self):
⋮
│    def start_processing(self):
⋮
│    def update_logs(self, line):
⋮
│    def update_progress_bar(self, progress):
⋮
│    def stop_processing(self):
⋮
│    def force_kill_if_needed(self):
⋮
│    def reset_ui(self):
⋮
│    def save_config(self):
⋮
│    def load_config(self):
⋮
│    def handle_error(self, error_message):
⋮
│    def closeEvent(self, event):
⋮
│    def check_ffmpeg_available(self):
⋮

setup.ps1

setup.sh

ui.yml

utils.py:
⋮
│colormap = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'red',
⋮
│model = None
│processor = None
│
│def set_model_info(model_, processor_):
⋮
│class TaskType(str, Enum):
⋮
│def run_example(task_prompt: TaskType, image, text_input=None):
⋮
│def draw_polygons(image, prediction, fill_mask=False):
⋮
│def draw_ocr_bboxes(image, prediction):
⋮
│def convert_bbox_to_relative(box, image):
⋮
│def convert_relative_to_bbox(relative, image):
⋮
│def convert_bbox_to_loc(box, image):
⋮

