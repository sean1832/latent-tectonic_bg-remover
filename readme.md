# Latent Tectonic: Background Removal Tool
This tool is build for batch processing of images to remove backgrounds using masks for Latent Tectonic elective.

## Features
- Batch background removal using masks
- Multi threaded
- Flexible mask modes (black, white, color)

## Requirements
- Python 3.10 or higher

## Installation
1. Clone the repo
```bash
git clone https://github.com/sean1832/latent-tectonic_bg-remover.git
```

2. Navigate to the project directory
```bash
cd latent-tectonic_bg-remover
```

3. Create and activate a virtual environment (optional but recommended)
```bash
python -m venv venv
./venv/bin/activate
```

4. Install the required packages
```bash
pip install -r requirements.txt
```



## Usage
Place your input images in the `input` directory and your mask images in the `mask` directory. Then run:
```bash
python remove-bg.py
```


### Other Examples

1. Use 8 threads, remove white background with threshold 20
```bash
python remove-bg.py --workers 8 --mask-mode white --threshold 20
```

2. Remove green screen (RGB 0,255,0) within tolerance 12
```bash
python remove-bg.py --mask-mode color --color 0,255,0 --color-tolerance 12
```

3. Disable renaming and cropping
   
```bash
python remove-bg.py --no-rename --no-crop
```
4 Custom input/output folders
```bash
python remove-bg.py --input photos --mask masks --output cleaned
```

### Commands

| Option                  | Type / Values               | Default  | Description                                                                                                                                                                      |     |
| ----------------------- | --------------------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- |
| `--input PATH`          | Path                        | `input`  | Directory with input images                                                                                                                                                      |     |
| `--mask PATH`           | Path                        | `mask`   | Directory with mask images                                                                                                                                                       |     |
| `--output PATH`         | Path                        | `output` | Directory where results are saved                                                                                                                                                |     |
| `--temp PATH`           | Path                        | `.temp`  | Temporary directory for safe renaming                                                                                                                                            |     |
| `--no-crop`             | Flag                        | -        | Disable center square cropping (enabled by default)                                                                                                                              |     |
| `--no-rename`           | Flag                        | -        | Disable automatic renaming/alignment of input and mask pairs                                                                                                                     |     |
| `--sequence-zero-pad N` | Integer                     | `3`      | Zero padding for renaming sequence (e.g. 001, 002, …)                                                                                                                            |     |
| `--workers N`           | Integer or `max`            | `max`    | Number of worker threads. Use `max` for all CPU cores                                                                                                                            |
| `--mask-mode MODE`      | `black` / `white` / `color` | `black`  | How to select the background to remove: <br> • **black** -> pixels ≤ threshold <br> • **white** -> pixels ≥ (255 − threshold) <br> • **color** -> pixels close to a target color |     |
| `--threshold N`         | Integer                     | `10`     | Threshold for black/white modes                                                                                                                                                  |     |
| `--color R,G,B`         | Integers                    | `0,0,0`  | Target color (only for `color` mode)                                                                                                                                             |     |
| `--color-tolerance N`   | Integer                     | `15`     | RGB distance tolerance for `color` mode                                                                                                                                          |     |